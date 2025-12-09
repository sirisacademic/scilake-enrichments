"""
RerankerLinker: Two-stage entity linking with retrieval + reranking.

Architecture:
    Stage 1 - Retrieval (embeddings):
        - Fast semantic search over taxonomy
        - Returns top-k candidates with similarity scores
        - Optional: adds top-level categories as fallbacks
    
    Stage 2 - Reranking (LLM):
        - LLM evaluates candidates with context
        - Can select best match or REJECT if not domain-relevant
        - Understands nuanced context and domain-specific terminology

Benefits:
    Fast: embedding retrieval is ~10-20ms per entity
    Accurate: LLM catches semantic nuances
    Safe: Can reject non-domain terms
    Flexible: Works with or without context
    Domain-agnostic: Works for any domain taxonomy
"""

import re
import numpy as np
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path


class RerankerLinker:
    def __init__(
        self, 
        taxonomy_path: str,
        domain: str = "energy",
        embedding_model_name: str = "intfloat/multilingual-e5-large-instruct",
        llm_model_name: str = "Qwen/Qwen3-1.7B",
        threshold: float = 0.7,
        context_window: int = 3,
        max_contexts: int = 3,
        use_sentence_context: bool = False,
        use_context_for_retrieval: bool = False,  # NEW PARAMETER
        top_k_candidates: int = 5,
        add_top_level_fallbacks: bool = True,
        enable_thinking: bool = False,
        logger=None
    ):
        """
        Initialize reranker linker with embedding for retrieval + LLM for reranking.
        
        Args:
            taxonomy_path: Path to taxonomy TSV file
            domain: Domain name (e.g., "energy", "biology", "medicine")
            embedding_model_name: embedding model for candidate retrieval
            llm_model_name: LLM for reranking/selection
            threshold: Minimum similarity for embedding retrieval
            context_window: Number of tokens around entity
            max_contexts: Maximum contexts to extract
            use_sentence_context: Use sentences vs token windows
            top_k_candidates: Number of candidates from embedding
            add_top_level_fallbacks: Add top-level categories as fallback options
            enable_thinking: Enable LLM thinking mode (slower)
            use_context_for_retrieval: Use context in embedding retrieval (stage 1)
                                       LLM reranking (stage 2) always uses context
            logger: Optional logger instance
        """
        self.domain = domain
        self.threshold = threshold
        self.context_window = context_window
        self.max_contexts = max_contexts
        self.use_sentence_context = use_sentence_context
        self.use_context_for_retrieval = use_context_for_retrieval
        self.top_k = top_k_candidates
        self.add_top_level_fallbacks = add_top_level_fallbacks
        self.enable_thinking = enable_thinking
        self.logger = logger
        
        # Load embedding model
        if self.logger:
            self.logger.info(f"🔍 Loading embedding model: {embedding_model_name}")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Load LLM
        if self.logger:
            self.logger.info(f"🤖 Loading LLM: {llm_model_name}")
        
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load spaCy
        if self.logger:
            self.logger.info("📖 Loading spaCy for context extraction")
        
        self.nlp = spacy.load("en_core_web_sm", disable=["ner"])
        
        # Build taxonomy index
        if self.logger:
            self.logger.info(f"🗃️ Building taxonomy index from {taxonomy_path}")
        
        self.taxonomy_index = self._build_taxonomy_index(taxonomy_path)
        
        if self.logger:
            self.logger.info(
                f"✅ Reranker linker ready: {len(self.taxonomy_index['metadata'])} entries\n"
                f"   Domain: {domain}\n"
                f"   Retrieval: {embedding_model_name}\n"
                f"   Reranking: {llm_model_name}\n"
                f"   Top-k: {top_k_candidates}, Fallbacks: {add_top_level_fallbacks}, Thinking: {enable_thinking}"
            )
    
    def _get_detailed_instruct(self, task_description: str, query: str) -> str:
        """Format query with instruction for embedding-instruct"""
        return f'Instruct: {task_description}\nQuery: {query}'
    
    def _parse_bool_value(self, value) -> bool:
        """Parse various boolean representations"""
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value > 0
        if isinstance(value, str):
            return value.strip().lower() in {'true', '1', 'y', 'yes'}
        return False
    
    def _build_taxonomy_index(self, taxonomy_path: str) -> Dict:
        """Build taxonomy index with embeddings"""
        
        def get_first(value):
            """Return the first element before a |, stripping whitespace."""
            if pd.isna(value):
                return None
            value = str(value).strip()
            if not value:
                return None
            return value.split("|")[0].strip()
        
        if not Path(taxonomy_path).is_absolute():
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        
        documents = []
        metadata = []
        top_level_ids = set()

        # Aliases
        aliases_cols = ["synonyms", "wikidata_aliases"]
        # Keep only the columns that actually exist in df
        existing = [c for c in aliases_cols if c in df.columns]
        # Create "aliases" by joining existing columns
        df["aliases"] = (
            df[existing]
            .fillna("")                      # replace NaN with empty string
            .apply(lambda row: " | ".join(
                [v for v in row if v]        # keep only non-empty values
            ), axis=1)
        )

        for _, row in df.iterrows():
            # Build rich document
            doc = row['concept']
            
            if row.get('description'):
                doc += f". {row['description']}"
            
            if row.get('aliases'):
                aliases = row['aliases'].replace(' | ', ', ')
                doc += f". Also known as: {aliases}"
            
            documents.append(doc)
            
            # Check if this is a top-level concept
            is_top_level = False
            if 'top_level' in row and pd.notna(row['top_level']):
                is_top_level = self._parse_bool_value(row['top_level'])
            
            if is_top_level:
                top_level_ids.add(str(row['id']))
            
            # Get parent_id (note: column might be 'parent-id' or 'parent_id')
            parent_id = None
                
            if 'parent-id' in row and pd.notna(row['parent-id']) and str(row['parent-id']).strip():
                parent_id = get_first(row['parent-id'])
            elif 'parent_id' in row and pd.notna(row['parent_id']) and str(row['parent_id']).strip():
                parent_id = get_first(row['parent_id'])
            
            metadata.append({
                'taxonomy_id': str(row['id']),
                'concept': row['concept'],
                'wikidata_id': row.get('wikidata_id', ''),
                'description': row.get('description', ''),
                'aliases': row.get('aliases', ''),
                'type': row.get('type', ''),
                'is_top_level': is_top_level,
                'parent_id': parent_id  # NEW: Store parent ID
            })
        
        if self.logger:
            self.logger.info(f"📊 Encoding {len(documents)} taxonomy documents...")
            self.logger.info(f"📂 Found {len(top_level_ids)} top-level categories")
        
        # Encode documents with embedding (no instruction prefix)
        embeddings = self.embedding_model.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32
        )
        
        return {
            'embeddings': np.array(embeddings),
            'metadata': metadata,
            'documents': documents,
            'top_level_ids': top_level_ids
        }
    
    def _extract_all_contexts(self, doc, entity_text: str) -> List[str]:
        """Extract contexts for entity occurrences"""
        entity_lower = entity_text.lower()
        contexts = []
        seen_positions = set()
        
        if self.use_sentence_context:
            for sent in doc.sents:
                if entity_lower in sent.text.lower():
                    if sent.start_char not in seen_positions:
                        seen_positions.add(sent.start_char)
                        contexts.append(sent.text.strip())
                        if len(contexts) >= self.max_contexts:
                            break
        else:
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
    
    def _is_top_level(self, taxonomy_id: str) -> bool:
        """Check if taxonomy ID is a top-level category"""
        return taxonomy_id in self.taxonomy_index['top_level_ids']
    
    def _add_top_level_fallbacks(
        self, 
        candidates: List[Tuple[str, float]], 
        all_scores: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Add top-level categories as fallback if not already present.
        Places them at the end of the candidate list.
        
        Args:
            candidates: List of (tid, score) from embedding retrieval
            all_scores: Full score array to get scores for top-level categories
            
        Returns:
            Extended candidate list with top-level fallbacks
        """
        # Get all top-level category IDs
        top_level_fallbacks = list(self.taxonomy_index['top_level_ids'])
        
        # Get set of already included taxonomy IDs
        existing_ids: Set[str] = {tid for tid, _ in candidates}
        
        # Add top-level categories that aren't already in candidates
        for fallback_id in top_level_fallbacks:
            if fallback_id not in existing_ids:
                # Find the index and score for this fallback
                try:
                    meta_idx = next(
                        i for i, m in enumerate(self.taxonomy_index['metadata']) 
                        if m['taxonomy_id'] == fallback_id
                    )
                    score = float(all_scores[meta_idx])
                    
                    # Add to end of candidates (lower priority than embedding results)
                    candidates.append((fallback_id, score))
                    
                    if self.logger:
                        meta = self.taxonomy_index['metadata'][meta_idx]
                        self.logger.debug(
                            f"   📂 Added fallback: {meta['concept']} ({fallback_id}, score={score:.3f})"
                        )
                except StopIteration:
                    # Fallback ID not found in taxonomy
                    if self.logger:
                        self.logger.warning(f"⚠️ Fallback ID {fallback_id} not found in taxonomy")
        
        return candidates
    
    def _get_candidates_with_embedding(
        self, 
        entity_text: str, 
        contexts: List[str]
    ) -> List[Tuple[str, float]]:
        """
        Step 1: Use embedding to retrieve top-k candidates.
        Optionally adds top-level categories as fallback options.
        
        Returns list of (taxonomy_id, score) tuples.
        """
        # Build query - optionally include context based on parameter
        if self.use_context_for_retrieval and contexts and self.max_contexts > 0:
            context_examples = "\n".join([f"- {ctx}" for ctx in contexts[:self.max_contexts]])
            query_text = f"{entity_text}\n\nExample occurrences:\n{context_examples}"
            task = f'Given the term "{entity_text}" with usage examples, retrieve the taxonomy entry that matches this term'
        else:
            # Entity text only - no context contamination
            query_text = entity_text
            task = f'Given the term "{entity_text}", retrieve the taxonomy entry that matches this term'
        
        # Add instruction
        query = self._get_detailed_instruct(task, query_text)
        
        # Encode query
        query_emb = self.embedding_model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute similarities
        scores = query_emb @ self.taxonomy_index['embeddings'].T
        
        # Get top-k from embedding retrieval
        top_indices = scores.argsort()[-self.top_k:][::-1]
        
        candidates = []
        for idx in top_indices:
            tid = self.taxonomy_index['metadata'][idx]['taxonomy_id']
            score = float(scores[idx])
            candidates.append((tid, score))
        
        # Add top-level categories if requested and not already present
        if self.add_top_level_fallbacks:
            candidates = self._add_top_level_fallbacks(candidates, scores)
        
        if self.logger:
            self.logger.debug(f"🔍 Embedding candidates for '{entity_text}':")
            for tid, score in candidates:
                meta = next(m for m in self.taxonomy_index['metadata'] if m['taxonomy_id'] == tid)
                marker = "📂" if self._is_top_level(tid) else "  "
                self.logger.debug(f"   {marker} {meta['concept']} ({tid}): {score:.3f}")
        
        return candidates

    # !!!!! TODO: UPDATE FOR SCILAKE DOMAINS !!!!!

    def _build_llm_prompt(
        self, 
        entity_text: str, 
        contexts: List[str], 
        candidates: List[Tuple[str, float]]
    ) -> str:
        """Build concise prompt for LLM reranking with entity highlighting"""
        
        import re
        
        # Format contexts with entity highlighted
        context_text = ""
        if contexts:
            for i, ctx in enumerate(contexts[:self.max_contexts], 1):
                # Highlight entity in context (case-insensitive replacement)
                highlighted_ctx = re.sub(
                    re.escape(entity_text), 
                    f">>>{entity_text}<<<", 
                    ctx, 
                    flags=re.IGNORECASE
                )
                context_text += f"{i}. {highlighted_ctx}\n"
        else:
            context_text = "(no context)\n"
        
        # Build parent-child relationships
        candidate_hierarchy = {}
        for tid, score in candidates:
            meta = next(m for m in self.taxonomy_index['metadata'] if m['taxonomy_id'] == tid)
            parent = meta.get('parent_id')
            if parent and parent in [c[0] for c in candidates]:
                if parent not in candidate_hierarchy:
                    candidate_hierarchy[parent] = []
                candidate_hierarchy[parent].append(tid)
        
        # Format candidates with simple markers
        candidate_text = ""
        for i, (tid, score) in enumerate(candidates, 1):
            meta = next(m for m in self.taxonomy_index['metadata'] if m['taxonomy_id'] == tid)
            
            marker = ""
            if self._is_top_level(tid):
                marker = " [BROAD]"
            elif tid in candidate_hierarchy:
                children = candidate_hierarchy[tid]
                child_positions = [j+1 for j, (c_tid, _) in enumerate(candidates) if c_tid in children]
                marker = f" [→see {child_positions}]"
            elif meta.get('parent_id') in [c[0] for c in candidates]:
                marker = " [SPECIFIC]"
            
            candidate_text += f"[{i}] {meta['concept']}{marker}\n"
        
        # Get domain guidance
        domain_guidance = self._get_domain_guidance()
        
        # Build prompt with highlighted entity
        prompt = f"""Match entity to best category.
Entity: >>>{entity_text}<

Categories:
{candidate_text}

Instructions:
Step 1: Evaluate the COMPLETE entity text
- Read the FULL entity text: >>>{entity_text}<
- Does the complete entity text (including all modifiers like "offshore", "onshore", etc.) match a specific category?
- If yes, choose that specific category

Step 2: Use context only if needed for disambiguation
Context (entity marked with >>>...<<<):
{context_text}

General rules:
- Match based on the ENTITY TEXT (marked with >>>...<<<) first and foremost
- The entity text alone determines the category - context provides supporting information only
- Use context only to clarify ambiguous terms (e.g., "offshore" vs "onshore")
- Do NOT let context override what the entity clearly is
- If entity text contradicts context associations, trust the entity text
- Prefer [SPECIFIC] over [BROAD] categories
{domain_guidance}

Answer (number or REJECT):"""

        return prompt

    def _get_domain_guidance(self) -> str:
        """Domain-specific matching rules"""
        
        guidance_map = {
            "energy": """
Domain rules (energy):
- Match ONLY: energy sources, fuels, generation technologies, storage methods
- REJECT: chemical substances, emissions, aerosols, pollutants, combustion byproducts

IMPORTANT:
- If entity is a chemical compound or atmospheric substance rather than an energy source/fuel, answer REJECT
    
""",
       
            "generic": """
Domain rules:
- Match terms used in the specific domain context
- REJECT terms from unrelated domains
- Prefer specific technical terms over general ones"""
        }
        
        return guidance_map.get(self.domain.lower(), guidance_map["generic"])


    def _query_llm(self, prompt: str) -> str:
        """Query LLM and return response"""
        messages = [{"role": "user", "content": prompt}]
        
        text = self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(self.llm_model.device)
        
        # Adjust max tokens based on thinking mode
        max_tokens = 150 if self.enable_thinking else 50
        
        with torch.no_grad():
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                eos_token_id=self.llm_tokenizer.eos_token_id,
                pad_token_id=self.llm_tokenizer.pad_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        answer = self.llm_tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return answer
        
    def _parse_llm_answer(
        self, 
        answer: str, 
        candidates: List[Tuple[str, float]]
    ) -> Optional[Tuple[str, float]]:
        """Parse LLM answer to get (taxonomy_id, score)"""
        answer_upper = answer.upper()
        
        if "REJECT" in answer_upper:
            return None
        
        # Try to extract number
        for char in answer:
            if char.isdigit():
                num = int(char)
                if 1 <= num <= len(candidates):
                    return candidates[num - 1]  # Returns (tid, score)
        
        return None
    
    def link_entity_with_contexts(
        self, 
        entity_text: str,
        contexts: List[str],
        taxonomy_source: str = "taxonomy"
    ) -> Optional[List[Dict]]:
        """
        Link entity using reranker approach: embedding retrieval + LLM reranking.
        """
        # Step 1: Get candidates from embedding
        candidates = self._get_candidates_with_embedding(entity_text, contexts)
        
        if not candidates:
            if self.logger:
                self.logger.debug(f"❌ No embedding candidates for '{entity_text}'")
            return None
        
        # Check if top candidate is below threshold
        if candidates[0][1] < self.threshold:
            if self.logger:
                self.logger.debug(
                    f"❌ '{entity_text}' below embedding threshold: {candidates[0][1]:.3f} < {self.threshold}"
                )
            return None
        
        # Step 2: Use LLM to rerank/select
        prompt = self._build_llm_prompt(entity_text, contexts, candidates)
        answer = self._query_llm(prompt)
        
        # Parse LLM answer
        result = self._parse_llm_answer(answer, candidates)
        
        # Build enriched candidate info for cache
        candidates_info = []
        for tid, score in candidates:
            meta = next(m for m in self.taxonomy_index['metadata'] if m['taxonomy_id'] == tid)
            candidates_info.append({
                'taxonomy_id': tid,
                'concept': meta['concept'],
                'score': float(score),
                'is_top_level': self._is_top_level(tid)
            })
        
        if not result:
            if self.logger:
                self.logger.debug(
                    f"❌ '{entity_text}' rejected by LLM (answer: {answer})"
                )
            # Return None for linking, but include debug info
            return None, {
                'candidates': candidates_info,
                'llm_answer': answer,
                'rejected': True
            }
        
        selected_id, embedding_score = result
        
        # Get match metadata
        meta = next(
            m for m in self.taxonomy_index['metadata'] 
            if m['taxonomy_id'] == selected_id
        )
        
        if self.logger:
            self.logger.debug(
                f"✅ '{entity_text}' → '{meta['concept']}' "
                f"({taxonomy_source}:{selected_id}, embedding={embedding_score:.3f}, LLM={answer})"
            )
        
        # Build linking result
        linking = [{
            'source': taxonomy_source,
            'id': selected_id,
            'name': meta['concept'],
            'score': embedding_score
        }]
        
        # Add Wikidata if available
        if meta['wikidata_id']:
            wd_id = meta['wikidata_id'].split('/')[-1]
            linking.append({
                'source': 'Wikidata',
                'id': wd_id,
                'name': meta['concept']
            })
        
        # Return linking + debug info
        debug_info = {
            'candidates': candidates_info,
            'llm_answer': answer,
            'rejected': False
        }
        
        return linking, debug_info
    
    def link_entities_in_section(
        self,
        section_text: str,
        entities: List[Dict],
        cache: Dict,
        taxonomy_source: str = "taxonomy",
        filename: str = None,
        section_id: str = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Link all entities in a section using reranker approach.
        """
        enriched = []
        cache_hits = 0
        cache_misses = 0
        links_added = 0
        skipped_invalid = 0
        
        for entity in entities:
            # Skip if already has linking (from gazetteer)
            if entity.get('linking'):
                enriched.append(entity)
                continue
            
            entity_text = entity['text'].lower()
            
            # Validate that entity actually appears in text
            if entity['text'].lower() not in section_text.lower():
                if self.logger:
                    # Build informative log message with available info
                    location_info = []
                    if filename:
                        location_info.append(f"file: {filename}")
                    if section_id:
                        location_info.append(f"section: {section_id}")
                    
                    location_str = f" ({', '.join(location_info)})" if location_info else ""
                    
                    self.logger.warning(
                        f"⚠️ Skipping invalid entity '{entity['text']}'{location_str} "
                        f"(not found in section text)"
                    )
                skipped_invalid += 1
                continue
            
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
            doc = self.nlp(section_text)
            contexts = self._extract_all_contexts(doc, entity['text'])
            contexts = list(dict.fromkeys(contexts))  # Deduplicate
            
            if self.logger:
                self.logger.debug(f"🔍 Found {len(contexts)} occurrences of '{entity['text']}'")
            
            # Link entity with reranker approach
            result = self.link_entity_with_contexts(entity['text'], contexts, taxonomy_source)
            
            if result is None:
                # No candidates or rejected
                linking = None
                debug_info = {'rejected': True, 'reason': 'no_candidates'}
            else:
                linking, debug_info = result
            
            # Cache result with debug info
            cache[entity_text] = {
                'linking': linking,
                'contexts': contexts,
                'debug': debug_info
            }
            cache_misses += 1
            
            # Add linking to entity
            entity_copy = entity.copy()
            if linking:
                entity_copy['linking'] = linking
                links_added += 1
            enriched.append(entity_copy)
        
        if self.logger:
            log_msg = (
                f"📊 Cache: {cache_hits} hits, {cache_misses} misses | "
                f"Links added: {links_added}/{len(entities)}"
            )
            if skipped_invalid > 0:
                log_msg += f" | Skipped invalid: {skipped_invalid}"
            self.logger.debug(log_msg)
        
        return enriched, cache
