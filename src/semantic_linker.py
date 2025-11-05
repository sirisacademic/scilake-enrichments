"""
Semantic entity linking using multilingual-e5-base.
Links NER entities to taxonomies via sentence context matching.
Generic implementation supporting any taxonomy in TSV format.
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import spacy
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class SemanticLinker:
    def __init__(
        self, 
        taxonomy_path: str,
        model_name: str = "intfloat/multilingual-e5-base",
        threshold: float = 0.6,
        logger=None
    ):
        """
        Initialize semantic linker with taxonomy.
        
        Args:
            taxonomy_path: Path to taxonomy TSV file
            model_name: Sentence transformer model
            threshold: Minimum similarity score for linking
            logger: Optional logger instance
        """
        self.threshold = threshold
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"🤖 Loading sentence transformer: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        
        if self.logger:
            self.logger.info("📝 Loading spaCy for sentence segmentation")
        
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        
        # Build in-memory taxonomy index
        if self.logger:
            self.logger.info(f"🏗️  Building taxonomy index from {taxonomy_path}")
        
        self.taxonomy_index = self._build_taxonomy_index_centroid(taxonomy_path)
        
        if self.logger:
            self.logger.info(
                f"✅ Taxonomy index ready: {len(self.taxonomy_index['metadata'])} entries "
                f"({self.taxonomy_index['embeddings'].shape[0]} embeddings)"
            )

    def _build_taxonomy_index_centroid(self, taxonomy_path: str) -> Dict:
        """
        Build in-memory index with embeddings for all taxonomy concepts + aliases.
        Combines concept and aliases into a single centroid embedding.
        
        Returns:
            {
                'embeddings': np.array [N, 768],
                'metadata': List[Dict] with taxonomy_id, matched_text, wikidata_id
            }
        """
        # Resolve path
        if not Path(taxonomy_path).is_absolute():
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        
        embeddings = []
        metadata = []
        
        for _, row in df.iterrows():
            # Primary concept
            concept = row['concept']
            wikidata_id = row.get('wikidata_id', '')
            
            # Encode the concept embedding (no prefix for symmetric similarity)
            concept_emb = self.model.encode([concept], normalize_embeddings=True)[0]
            
            # Collect alias embeddings
            alias_embeddings = [concept_emb]  # Start with the concept embedding
            aliases = row.get('wikidata_aliases')
            if pd.notna(aliases) and isinstance(aliases, str):
                for alias in aliases.split(' | '):
                    alias = alias.strip()
                    if alias:
                        # Generate embedding for alias (no prefix)
                        alias_emb = self.model.encode([alias], normalize_embeddings=True)[0]
                        alias_embeddings.append(alias_emb)
            
            # Compute the centroid (average) of the concept and its aliases
            centroid_emb = np.mean(alias_embeddings, axis=0)
            embeddings.append(centroid_emb)
            
            # Store only one metadata entry per concept
            metadata.append({
                'taxonomy_id': str(row['id']),
                'matched_text': concept,
                'wikidata_id': wikidata_id,
                'type': row.get('type', '')
            })
        
        if self.logger:
            self.logger.info(f"📊 Built index: {len(metadata)} concepts with centroid embeddings")
        
        return {
            'embeddings': np.array(embeddings),
            'metadata': metadata
        }

    def _extract_all_contexts(self, doc, entity_text: str, use_sentence: bool = False, token_window: int = 3) -> List[str]:
        """
        Extract contexts for all unique occurrences of entity (handles multi-word entities).
        
        Args:
            doc: spaCy Doc object
            entity_text: Entity surface form (can be multi-word)
            use_sentence: If True, extract full sentences; if False, use token windows
            token_window: Number of tokens around entity (only used when use_sentence=False)
            
        Returns:
            List of unique contexts containing the entity
        """
        entity_lower = entity_text.lower()
        contexts = []
        seen_positions = set()
        
        if use_sentence:
            # Simple sentence-based extraction: find sentences containing the entity text
            for sent in doc.sents:
                if entity_lower in sent.text.lower():
                    # Check if we've already processed this sentence position
                    if sent.start_char not in seen_positions:
                        seen_positions.add(sent.start_char)
                        contexts.append(sent.text.strip())
        else:
            # Token window extraction: find exact token sequences
            entity_tokens = entity_lower.split()
            entity_len = len(entity_tokens)
            
            # Slide through doc with a window matching entity length
            for i in range(len(doc) - entity_len + 1):
                # Check if tokens match the entity
                window_tokens = [doc[i + j].text.lower() for j in range(entity_len)]
                
                if window_tokens == entity_tokens:
                    # Get character position of first token
                    char_pos = doc[i].idx
                    
                    if char_pos not in seen_positions:
                        seen_positions.add(char_pos)
                        
                        # Extract token window around entity
                        start_idx = max(0, i - token_window)
                        end_idx = min(len(doc), i + entity_len + token_window)
                        context = " ".join([t.text for t in doc[start_idx:end_idx]])
                        contexts.append(context)
        
        return contexts

    def link_entity_all_contexts(
        self, 
        entity_text: str, 
        contexts: List[str],
        taxonomy_source: str = "Taxonomy"
    ) -> Optional[List[Dict]]:
        """
        Link entity using centroid of all context embeddings (symmetric similarity).
        
        Args:
            entity_text: Entity surface form
            contexts: List of contexts (strings) containing the entity
            taxonomy_source: Name of taxonomy source for linking metadata (e.g., "IRENA", "UBERON")
            
        Returns:
            List of linking dicts, or None if below threshold
        """
        if not contexts:
            return None
        
        # Generate embeddings for all contexts (no prefix - symmetric similarity)
        context_embeddings = []
        for context in contexts:
            context_emb = self.model.encode(
                context,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            context_embeddings.append(context_emb)
        
        # Compute centroid by averaging the embeddings
        centroid_emb = np.mean(context_embeddings, axis=0)
        
        # Compute similarity with taxonomy index
        scores = centroid_emb @ self.taxonomy_index['embeddings'].T
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        
        if best_score < self.threshold:
            if self.logger:
                self.logger.debug(
                    f"❌ '{entity_text}' below threshold: {best_score:.3f} < {self.threshold}"
                )
            return None
        
        match = self.taxonomy_index['metadata'][best_idx]
        
        if self.logger:
            self.logger.debug(
                f"✅ '{entity_text}' → '{match['matched_text']}' "
                f"({taxonomy_source}:{match['taxonomy_id']}, score={best_score:.3f})"
            )
        
        # Build linking structure
        linking = [{
            'source': taxonomy_source,
            'id': match['taxonomy_id'],
            'name': match['matched_text'],
            'score': best_score
        }]
        
        # Add Wikidata if available
        if match['wikidata_id'] and isinstance(match['wikidata_id'], str):
            wd_id = match['wikidata_id'].split('/')[-1]
            linking.append({
                'source': 'Wikidata',
                'id': wd_id,
                'name': match['matched_text']
            })
        
        return linking

    def link_entities_in_section(
        self,
        section_text: str,
        entities: List[Dict],
        cache: Dict,
        taxonomy_source: str = "Taxonomy"
    ) -> Tuple[List[Dict], Dict]:
        """
        Link all entities in a section, using cache.
        
        Args:
            section_text: Full section text
            entities: List of entity dicts from NER
            cache: Linking cache {entity_text: {linking, contexts}}
            taxonomy_source: Name of taxonomy source (e.g., "IRENA", "UBERON")
            
        Returns:
            (enriched_entities, updated_cache)
        """
        enriched = []
        cache_hits = 0
        cache_misses = 0
        links_added = 0
        
        for entity in entities:
            # Skip if already has linking (from gazetteer)
            if entity.get('linking'):
                enriched.append(entity)
                continue
            
            entity_text = entity['text'].lower()
            
            # Check cache
            if entity_text in cache:
                if self.logger:
                    self.logger.debug(f"💾 Cache hit for '{entity_text}'")
                    
                entity_copy = entity.copy()
                entity_copy['linking'] = cache[entity_text]['linking']
                enriched.append(entity_copy)
                cache_hits += 1
                if entity_copy.get('linking'):
                    links_added += 1
                continue
            
            # Extract all contexts for this entity in the section text
            doc = self.nlp(section_text)
            contexts = self._extract_all_contexts(doc, entity['text'], use_sentence=False)

            # Deduplicate while preserving order
            contexts = list(dict.fromkeys(contexts))

            if self.logger:
                self.logger.debug(f"📍 Found {len(contexts)} occurrences of '{entity['text']}'")

            if not contexts:
                if self.logger:
                    self.logger.debug(f"⚠️  Could not extract context for '{entity['text']}'")
                enriched.append(entity)
                cache_misses += 1
                continue
            
            # Link using centroid of all contexts
            linking = self.link_entity_all_contexts(entity['text'], contexts, taxonomy_source)

            # Cache result (even if None)
            cache[entity_text] = {
                'linking': linking,
                'contexts': contexts
            }
            
            cache_misses += 1
            
            # Add linking to entity
            entity_copy = entity.copy()
            if linking:
                entity_copy['linking'] = linking
                links_added += 1
            enriched.append(entity_copy)
        
        if self.logger:
            self.logger.debug(
                f"📊 Cache: {cache_hits} hits, {cache_misses} misses | "
                f"Links added: {links_added}/{len(entities)}"
            )
        
        return enriched, cache
