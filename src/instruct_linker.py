"""
Entity linking using instruction-based embedding models.
Uses instruction-based retrieval: queries have instructions, documents don't.
Compatible with models like multilingual-e5-large-instruct.
"""

import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Tuple
from pathlib import Path


class InstructLinker:
    def __init__(
        self, 
        taxonomy_path: str,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        threshold: float = 0.7,
        context_window: int = 3,
        max_contexts: int = 3,
        use_sentence_context: bool = False,
        logger=None
    ):
        """
        Initialize instruction-based linker with taxonomy.
        
        Args:
            taxonomy_path: Path to taxonomy TSV file
            model_name: Instruction-based embedding model
            threshold: Minimum similarity score for linking
            context_window: Number of tokens around entity (for token-based contexts)
            max_contexts: Maximum number of contexts to extract (sentences or token windows)
            use_sentence_context: If True, use full sentences; if False, use token windows
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.threshold = threshold
        self.context_window = context_window
        self.max_contexts = max_contexts
        self.use_sentence_context = use_sentence_context
        self.logger = logger
        
        if self.logger:
            self.logger.info(f"🤖 Loading model: {model_name}")
        
        self.model = SentenceTransformer(model_name)
        
        if self.logger:
            self.logger.info("📖 Loading spaCy for context extraction")
        
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        
        if self.logger:
            self.logger.info(f"🗃️ Building taxonomy index from {taxonomy_path}")
        
        self.taxonomy_index = self._build_taxonomy_index(taxonomy_path)
        
        if self.logger:
            self.logger.info(
                f"✅ Taxonomy ready: {len(self.taxonomy_index['metadata'])} entries "
                f"(context_window={context_window}, max_contexts={max_contexts}, "
                f"sentence_context={use_sentence_context})"
            )
    
    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with instruction (e5-instruct format)"""
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _build_taxonomy_index(self, taxonomy_path: str) -> Dict:
        """
        Build taxonomy index with rich documents (NO instruction prefix).
        Documents include: concept + description + aliases.
        """
        # Resolve path
        if not Path(taxonomy_path).is_absolute():
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        
        documents = []
        metadata = []
        
        for _, row in df.iterrows():
            # Build rich document
            doc = row['concept']
            
            # Add description if available
            if row.get('description'):
                doc += f". {row['description']}"
            
            # Add aliases if available
            if row.get('wikidata_aliases'):
                aliases = row['wikidata_aliases'].replace(' | ', ', ')
                doc += f". Also known as: {aliases}"
            
            documents.append(doc)
            
            metadata.append({
                'taxonomy_id': str(row['id']),
                'concept': row['concept'],
                'wikidata_id': row.get('wikidata_id', ''),
                'type': row.get('type', '')
            })
        
        if self.logger:
            self.logger.info(f"📊 Encoding {len(documents)} taxonomy documents...")
        
        # Encode documents WITHOUT instruction
        embeddings = self.model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        return {
            'embeddings': np.array(embeddings),
            'metadata': metadata
        }
    
    def _extract_all_contexts(self, doc, entity_text: str) -> List[str]:
        """
        Extract contexts for entity occurrences (handles multi-word entities).
        
        Args:
            doc: spaCy Doc object
            entity_text: Entity surface form (can be multi-word)
            
        Returns:
            List of unique contexts containing the entity (limited to max_contexts)
        """
        entity_lower = entity_text.lower()
        contexts = []
        seen_positions = set()
        
        if self.use_sentence_context:
            # Sentence-based extraction
            for sent in doc.sents:
                if entity_lower in sent.text.lower():
                    if sent.start_char not in seen_positions:
                        seen_positions.add(sent.start_char)
                        contexts.append(sent.text.strip())
                        if len(contexts) >= self.max_contexts:
                            break
        else:
            # Token window extraction
            entity_tokens = entity_lower.split()
            entity_len = len(entity_tokens)
            
            for i in range(len(doc) - entity_len + 1):
                window_tokens = [doc[i + j].text.lower() for j in range(entity_len)]
                
                if window_tokens == entity_tokens:
                    char_pos = doc[i].idx
                    
                    if char_pos not in seen_positions:
                        seen_positions.add(char_pos)
                        
                        start_idx = max(0, i - self.context_window)
                        end_idx = min(len(doc), i + entity_len + self.context_window)
                        context = " ".join([t.text for t in doc[start_idx:end_idx]])
                        contexts.append(context)
                        if len(contexts) >= self.max_contexts:
                            break
        
        return contexts
    
    def link_entity_with_contexts(
        self, 
        entity_text: str,
        contexts: List[str],
        taxonomy_source: str = "IRENA"
    ) -> Optional[List[Dict]]:
        """
        Link entity using contexts (if available).
        
        Args:
            entity_text: Entity surface form
            contexts: List of context strings
            taxonomy_source: Name of taxonomy (e.g., "IRENA")
            
        Returns:
            List of linking dicts, or None if below threshold
        """
        # Build query text
        if contexts and self.max_contexts > 0:
            # Limit to self.max_contexts contexts to avoid exceeding token limit
            context_examples = "\n".join([f"- {ctx}" for ctx in contexts[:self.max_contexts]])
            query_text = f"{entity_text}\n\nExample occurrences:\n{context_examples}"
            task = f'Given the term "{entity_text}" with usage examples, retrieve the taxonomy entry that matches this term'
        else:
            query_text = entity_text
            task = f'Given the term "{entity_text}", retrieve the taxonomy entry that matches this term'
        
        # Add instruction
        query = self._get_detailed_instruct(task, query_text)
        
        # Encode query
        query_emb = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute similarities
        scores = query_emb @ self.taxonomy_index['embeddings'].T
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        
        # Check threshold
        if best_score < self.threshold:
            if self.logger:
                self.logger.debug(
                    f"❌ '{entity_text}' below threshold: {best_score:.3f} < {self.threshold}"
                )
            return None
        
        match = self.taxonomy_index['metadata'][best_idx]
        
        if self.logger:
            self.logger.debug(
                f"✅ '{entity_text}' → '{match['concept']}' "
                f"({taxonomy_source}:{match['taxonomy_id']}, score={best_score:.3f})"
            )
        
        # Build linking structure
        linking = [{
            'source': taxonomy_source,
            'id': match['taxonomy_id'],
            'name': match['concept'],
            'score': best_score
        }]
        
        # Add Wikidata if available
        if match['wikidata_id']:
            wd_id = match['wikidata_id'].split('/')[-1]
            linking.append({
                'source': 'Wikidata',
                'id': wd_id,
                'name': match['concept']
            })
        
        return linking
    
    def link_entities_in_section(
        self,
        section_text: str,
        entities: List[Dict],
        cache: Dict,
        taxonomy_source: str = "IRENA"
    ) -> Tuple[List[Dict], Dict]:
        """
        Link all entities in a section, using cache.
        
        Args:
            section_text: Full section text
            entities: List of entity dicts from NER
            cache: Linking cache {entity_text: {linking, contexts}}
            taxonomy_source: Name of taxonomy (e.g., "IRENA")
            
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
            
            # Extract contexts
            contexts = []
            doc = self.nlp(section_text)
            contexts = self._extract_all_contexts(doc, entity['text'])
            contexts = list(dict.fromkeys(contexts))  # Deduplicate
            
            if self.logger:
                self.logger.debug(f"🔍 Found {len(contexts)} occurrences of '{entity['text']}'")
            
            # Link entity
            linking = self.link_entity_with_contexts(entity['text'], contexts, taxonomy_source)
            
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
