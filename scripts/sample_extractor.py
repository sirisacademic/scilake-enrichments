#!/usr/bin/env python3
"""
Extract diverse samples from NER+EL output for expert annotation.

Supports:
- Directory-level input (auto-matches sections CSV with NER JSONL files)
- File sampling weighted by size
- Entity sampling with diversity strategies
- Including unlinked entities for annotation
- Filtering by entity type (for cancer domain per-type sampling)

Outputs TSV with columns:
- id: section_id + offset of mention
- mention: surface text
- context: sentence or context window
- linked_to_concept_id: taxonomy entity ID ('[UNLINKED]' for unlinked)
- linked_to_name: taxonomy entity name (empty for unlinked)
- entity_type: NER-predicted entity type (e.g., Gene, Disease, Species)
- source_file: source document identifier
"""

import json
import pandas as pd
import spacy
import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import random
import numpy as np
from tqdm import tqdm


class SampleExtractor:
    def __init__(
        self,
        sections_dir: Optional[str] = None,
        ner_output_dir: Optional[str] = None,
        sections_path: Optional[str] = None,
        ner_output_path: Optional[str] = None,
        context_window: int = 100,
        use_sentence_context: bool = True,
        random_seed: int = 42,
        include_unlinked: bool = False,
        unlinked_ratio: float = 0.2
    ):
        """
        Initialize the sample extractor.
        
        Args:
            sections_dir: Directory containing sections CSVs (for batch mode)
            ner_output_dir: Directory containing NER JSONL files (for batch mode)
            sections_path: Single sections CSV path (for single-file mode)
            ner_output_path: Single NER JSONL path (for single-file mode)
            context_window: Character window for context (if not using sentences)
            use_sentence_context: Whether to extract sentence-level context
            random_seed: Random seed for reproducibility
            include_unlinked: Whether to include unlinked entities
            unlinked_ratio: Target ratio of unlinked entities (default: 0.2 = 20%)
        """
        self.context_window = context_window
        self.use_sentence_context = use_sentence_context
        self.random_seed = random_seed
        self.include_unlinked = include_unlinked
        self.unlinked_ratio = unlinked_ratio
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Store paths
        self.sections_dir = Path(sections_dir) if sections_dir else None
        self.ner_output_dir = Path(ner_output_dir) if ner_output_dir else None
        
        # Load spaCy for sentence segmentation
        print("📦 Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        
        # Discover and index files if directories provided
        if self.sections_dir and self.ner_output_dir:
            self.file_pairs = self._discover_file_pairs()
            print(f"✅ Found {len(self.file_pairs)} matched file pairs")
        elif sections_path and ner_output_path:
            # Single file mode - wrap in same structure
            doc_id = self._extract_doc_id(ner_output_path)
            self.file_pairs = [{
                'doc_id': doc_id,
                'sections_path': sections_path,
                'ner_path': ner_output_path,
                'ner_size': os.path.getsize(ner_output_path)
            }]
            print(f"✅ Single file mode: {doc_id}")
        else:
            raise ValueError("Provide either (sections_dir + ner_output_dir) or (sections_path + ner_output_path)")

    def _extract_doc_id(self, path: str) -> str:
        """Extract document ID from filename."""
        fname = Path(path).stem
        if fname.endswith('_sections'):
            fname = fname[:-9]
        return fname

    def _discover_file_pairs(self) -> List[Dict[str, Any]]:
        """Discover and match sections CSV files with NER JSONL files."""
        sections_index = {}
        for f in self.sections_dir.glob("*_sections.csv"):
            doc_id = self._extract_doc_id(str(f))
            sections_index[doc_id] = f
        
        file_pairs = []
        for f in self.ner_output_dir.glob("*.jsonl"):
            doc_id = self._extract_doc_id(str(f))
            if doc_id in sections_index:
                file_pairs.append({
                    'doc_id': doc_id,
                    'sections_path': str(sections_index[doc_id]),
                    'ner_path': str(f),
                    'ner_size': f.stat().st_size
                })
        
        file_pairs.sort(key=lambda x: x['doc_id'])
        
        print(f"   Found {len(sections_index)} sections files")
        print(f"   Found {len(list(self.ner_output_dir.glob('*.jsonl')))} NER files")
        print(f"   Matched {len(file_pairs)} pairs")
        
        return file_pairs

    def _has_valid_linking(self, entity: Dict) -> bool:
        """Check if entity has valid (non-null, non-empty) linking."""
        linking = entity.get('linking')
        if linking is None:
            return False
        if isinstance(linking, list) and len(linking) == 0:
            return False
        return True

    def _load_sections_for_file(self, sections_path: str) -> Dict[str, str]:
        """Load sections CSV and build section_id -> text mapping."""
        try:
            df = pd.read_csv(sections_path, dtype=str, engine='python')
            
            # Handle different column naming
            text_col = None
            for col in ['section_content_expanded', 'section_content', 'text', 'content']:
                if col in df.columns:
                    text_col = col
                    break
            
            if text_col is None:
                return {}
            
            id_col = 'section_id' if 'section_id' in df.columns else df.columns[0]
            
            return dict(zip(df[id_col].fillna(''), df[text_col].fillna('')))
        except Exception as e:
            print(f"   ⚠️  Error loading sections: {e}")
            return {}

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        sentence_boundaries: Optional[List[Tuple[int, int, str]]] = None
    ) -> str:
        """Extract context around entity mention."""
        if self.use_sentence_context and sentence_boundaries:
            for sent_start, sent_end, sent_text in sentence_boundaries:
                if sent_start <= start and end <= sent_end:
                    return sent_text.strip()
        
        # Fallback to character window
        ctx_start = max(0, start - self.context_window)
        ctx_end = min(len(text), end + self.context_window)
        return text[ctx_start:ctx_end].strip()

    def _generate_sample_id(self, section_id: str, start: int, end: int) -> str:
        """Generate unique sample ID."""
        if '#' in section_id:
            parts = section_id.split('/')[-1]
        else:
            parts = hashlib.md5(section_id.encode()).hexdigest()[:8]
        
        return f"{parts}#{start}-{end}"

    def _extract_samples_from_ner(
        self,
        ner_data: List[Dict[str, Any]],
        sections_map: Dict[str, str],
        doc_id: str
    ) -> List[Dict[str, Any]]:
        """Extract samples from a single file's NER output."""
        samples = []
        
        # Pre-compute sentence boundaries for each section
        sentence_cache = {}
        if self.use_sentence_context:
            for section_id, text in sections_map.items():
                doc = self.nlp(text)
                sentence_cache[section_id] = [
                    (sent.start_char, sent.end_char, sent.text)
                    for sent in doc.sents
                ]
        
        for record in ner_data:
            section_id = record.get('section_id', '')
            entities = record.get('entities', [])
            section_text = sections_map.get(section_id, '')
            
            if not section_text:
                continue
            
            for entity in entities:
                mention_from_ner = entity.get('text', '')
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                linking_model = entity.get('model', 'unknown')
                
                # Get NER-predicted entity type (e.g., "Gene", "Disease", "Species")
                # This is used for ALL entities (linked and unlinked)
                entity_type = entity.get('entity', '')
                
                # Check if linked
                is_linked = self._has_valid_linking(entity)
                
                # Extract linking info
                if is_linked:
                    linking = entity.get('linking', [])
                    primary_link = linking[0] if linking else {}
                    
                    linked_id = primary_link.get('id', '')
                    linked_name = primary_link.get('name', '')
                    
                    # Wikidata info
                    wikidata_link = linking[1] if len(linking) > 1 and linking[1].get('source') == 'Wikidata' else {}
                    wikidata_id = wikidata_link.get('id', '')
                    wikidata_name = wikidata_link.get('name', '')
                else:
                    linked_id = '[UNLINKED]'
                    linked_name = ''
                    wikidata_id = ''
                    wikidata_name = ''
                
                # Validate mention against text
                if start < 0 or end > len(section_text):
                    continue
                
                mention_from_text = section_text[start:end]
                offset_mismatch = mention_from_ner.lower() != mention_from_text.lower()
                
                if offset_mismatch:
                    continue
                
                # Extract context
                context = self._extract_context(
                    section_text, start, end, 
                    sentence_cache.get(section_id)
                )
                
                # Generate sample ID
                sample_id = self._generate_sample_id(section_id, start, end)
                
                samples.append({
                    'id': sample_id,
                    'mention': mention_from_ner,
                    'mention_from_text': mention_from_text,
                    'offset_mismatch': offset_mismatch,
                    'context': context,
                    'linking_model': linking_model,
                    'linked_to_concept_id': linked_id,
                    'linked_to_name': linked_name,
                    'entity_type': entity_type,  # Always NER-predicted type
                    'linked_to_wikidata_id': wikidata_id,
                    'linked_to_wikidata_name': wikidata_name,
                    'source_file': doc_id,
                    'is_linked': is_linked
                })
        
        return samples

    def extract_samples_from_files(
        self,
        file_pairs: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> pd.DataFrame:
        """Extract samples from a list of file pairs."""
        all_samples = []
        
        iterator = tqdm(file_pairs, desc="Extracting samples") if show_progress else file_pairs
        
        for fp in iterator:
            sections_map = self._load_sections_for_file(fp['sections_path'])
            
            if not sections_map:
                continue
            
            try:
                ner_data = []
                with open(fp['ner_path'], 'r') as f:
                    for line in f:
                        if line.strip():
                            ner_data.append(json.loads(line))
                
                samples = self._extract_samples_from_ner(
                    ner_data, sections_map, fp['doc_id']
                )
                all_samples.extend(samples)
            except Exception as e:
                print(f"   ⚠️  Error processing {fp['doc_id']}: {e}")
                continue
        
        return pd.DataFrame(all_samples)

    def sample_files(
        self,
        n_files: int,
        strategy: str = 'weighted',
        model_stats_df: Optional[pd.DataFrame] = None
    ) -> List[Dict[str, Any]]:
        """Sample files for processing."""
        if n_files >= len(self.file_pairs) or strategy == 'all':
            return self.file_pairs
        
        if strategy == 'uniform':
            return random.sample(self.file_pairs, n_files)
        
        elif strategy == 'model_balanced' and model_stats_df is not None:
            return self._sample_files_model_balanced(n_files, model_stats_df)
        
        else:  # weighted by file size
            sizes = [fp['ner_size'] for fp in self.file_pairs]
            total_size = sum(sizes)
            weights = [s / total_size for s in sizes]
            
            indices = np.random.choice(
                len(self.file_pairs),
                size=n_files,
                replace=False,
                p=weights
            )
            return [self.file_pairs[i] for i in indices]

    def _sample_files_model_balanced(
        self,
        n_files: int,
        model_stats_df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Sample files to balance model representation."""
        doc_models = model_stats_df.groupby('doc_id')['model'].apply(set).to_dict()
        
        models = model_stats_df['model'].unique().tolist()
        files_per_model = max(1, n_files // len(models))
        
        selected_docs = set()
        
        for model in models:
            model_docs = [
                doc for doc, doc_models_set in doc_models.items()
                if model in doc_models_set and doc not in selected_docs
            ]
            
            n_to_select = min(files_per_model, len(model_docs))
            if n_to_select > 0:
                selected = random.sample(model_docs, n_to_select)
                selected_docs.update(selected)
        
        remaining = n_files - len(selected_docs)
        if remaining > 0:
            available = [fp['doc_id'] for fp in self.file_pairs if fp['doc_id'] not in selected_docs]
            if available:
                additional = random.sample(available, min(remaining, len(available)))
                selected_docs.update(additional)
        
        return [fp for fp in self.file_pairs if fp['doc_id'] in selected_docs]

    def scan_model_distribution(
        self,
        file_pairs: Optional[List[Dict[str, Any]]] = None,
        show_progress: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Scan files to get model distribution."""
        if file_pairs is None:
            file_pairs = self.file_pairs
        
        rows = []
        global_counts = defaultdict(int)
        
        iterator = tqdm(file_pairs, desc="Scanning models") if show_progress else file_pairs
        
        for fp in iterator:
            ner_path = fp['ner_path']
            doc_id = fp['doc_id']
            
            model_counts = defaultdict(int)
            try:
                with open(ner_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            for ent in record.get('entities', []):
                                model = ent.get('model', 'unknown')
                                model_counts[model] += 1
                                global_counts[model] += 1
            except Exception:
                continue
            
            for model, count in model_counts.items():
                rows.append({
                    'doc_id': doc_id,
                    'model': model,
                    'count': count
                })
        
        return pd.DataFrame(rows), dict(global_counts)

    def select_diverse_samples(
        self,
        df: pd.DataFrame,
        n_samples: int = 100,
        strategy: str = 'combined',
        max_per_mention: int = 3
    ) -> pd.DataFrame:
        """Select diverse samples for annotation, optionally including unlinked entities."""
        if self.include_unlinked and 'is_linked' in df.columns:
            return self._select_with_unlinked(df, n_samples, strategy, max_per_mention)
        
        # Original behavior - only linked entities
        linked_df = df[df['is_linked'] == True] if 'is_linked' in df.columns else df
        
        if strategy == 'random':
            return linked_df.sample(n=min(n_samples, len(linked_df)), random_state=self.random_seed)
        elif strategy == 'stratified':
            return self._stratified_sample(linked_df, 'entity_type', n_samples)
        elif strategy == 'mention_diverse':
            return self._mention_diverse_sample(linked_df, n_samples, max_per_mention)
        elif strategy == 'model_stratified':
            return self._model_stratified_sample(linked_df, n_samples, max_per_mention)
        else:  # combined
            return self._combined_sample(linked_df, n_samples, max_per_mention)

    def _select_with_unlinked(
        self,
        df: pd.DataFrame,
        n_samples: int,
        strategy: str,
        max_per_mention: int = 3
    ) -> pd.DataFrame:
        """Select samples including both linked and unlinked entities."""
        linked_df = df[df['is_linked'] == True].copy()
        unlinked_df = df[df['is_linked'] == False].copy()
        
        print(f"\n📊 Entity pool: {len(linked_df)} linked, {len(unlinked_df)} unlinked")
        
        n_unlinked_target = int(n_samples * self.unlinked_ratio)
        n_linked_target = n_samples - n_unlinked_target
        
        if len(unlinked_df) < n_unlinked_target:
            n_unlinked_target = len(unlinked_df)
            n_linked_target = n_samples - n_unlinked_target
            print(f"   ⚠️  Only {len(unlinked_df)} unlinked entities available")
        
        print(f"   Target: {n_linked_target} linked + {n_unlinked_target} unlinked")
        
        if strategy == 'model_stratified':
            linked_selected = self._model_stratified_sample(linked_df, n_linked_target, max_per_mention)
        elif strategy == 'stratified':
            linked_selected = self._stratified_sample(linked_df, 'entity_type', n_linked_target)
        elif strategy == 'mention_diverse':
            linked_selected = self._mention_diverse_sample(linked_df, n_linked_target, max_per_mention)
        else:
            linked_selected = self._combined_sample(linked_df, n_linked_target, max_per_mention)
        
        unlinked_selected = self._select_diverse_unlinked(unlinked_df, n_unlinked_target, max_per_mention)
        
        result = pd.concat([linked_selected, unlinked_selected], ignore_index=True)
        result = result.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        print(f"   ✅ Selected {len(linked_selected)} linked + {len(unlinked_selected)} unlinked = {len(result)} total")
        
        return result

    def _select_diverse_unlinked(
        self,
        df: pd.DataFrame,
        n_samples: int,
        max_per_mention: int = 2
    ) -> pd.DataFrame:
        """Select diverse unlinked entities."""
        if len(df) == 0 or n_samples == 0:
            return pd.DataFrame()
        
        df = df.copy()
        df['mention_lower'] = df['mention'].str.lower()
        
        selected = []
        for mention_lower, group in df.groupby('mention_lower'):
            n_from_group = min(max_per_mention, len(group))
            selected.append(group.sample(n=n_from_group, random_state=self.random_seed))
        
        if not selected:
            return df.sample(n=min(n_samples, len(df)), random_state=self.random_seed)
        
        limited = pd.concat(selected, ignore_index=True)
        
        if len(limited) <= n_samples:
            return limited
        
        return limited.sample(n=n_samples, random_state=self.random_seed)

    def _stratified_sample(self, df: pd.DataFrame, stratify_col: str, n_samples: int) -> pd.DataFrame:
        """Stratified sampling by column."""
        if len(df) == 0:
            return df
        
        if stratify_col not in df.columns:
            return df.sample(n=min(n_samples, len(df)), random_state=self.random_seed)
        
        groups = df.groupby(stratify_col)
        n_groups = len(groups)
        per_group = max(1, n_samples // n_groups)
        
        samples = []
        for _, group in groups:
            n = min(per_group, len(group))
            samples.append(group.sample(n=n, random_state=self.random_seed))
        
        result = pd.concat(samples, ignore_index=True)
        
        if len(result) < n_samples:
            remaining = df[~df.index.isin(result.index)]
            if len(remaining) > 0:
                additional = remaining.sample(
                    n=min(n_samples - len(result), len(remaining)),
                    random_state=self.random_seed
                )
                result = pd.concat([result, additional], ignore_index=True)
        
        return result.head(n_samples)

    def _mention_diverse_sample(
        self,
        df: pd.DataFrame,
        n_samples: int,
        max_per_mention: int = 3
    ) -> pd.DataFrame:
        """Sample with diversity in mentions."""
        if len(df) == 0:
            return df
        
        df = df.copy()
        df['mention_lower'] = df['mention'].str.lower()
        
        samples = []
        for _, group in df.groupby('mention_lower'):
            n = min(max_per_mention, len(group))
            samples.append(group.sample(n=n, random_state=self.random_seed))
        
        result = pd.concat(samples, ignore_index=True)
        
        if len(result) > n_samples:
            result = result.sample(n=n_samples, random_state=self.random_seed)
        
        return result

    def _model_stratified_sample(
        self,
        df: pd.DataFrame,
        n_samples: int,
        max_per_mention: int = 3
    ) -> pd.DataFrame:
        """Stratified sampling by linking model, then diverse by mention."""
        if len(df) == 0:
            return df
        
        if 'linking_model' not in df.columns:
            return self._mention_diverse_sample(df, n_samples, max_per_mention)
        
        models = df['linking_model'].unique()
        per_model = max(1, n_samples // len(models))
        
        samples = []
        for model in models:
            model_df = df[df['linking_model'] == model]
            model_samples = self._mention_diverse_sample(model_df, per_model, max_per_mention)
            samples.append(model_samples)
        
        result = pd.concat(samples, ignore_index=True)
        
        if len(result) < n_samples:
            remaining = df[~df.index.isin(result.index)]
            if len(remaining) > 0:
                additional = self._mention_diverse_sample(
                    remaining,
                    n_samples - len(result),
                    max_per_mention
                )
                result = pd.concat([result, additional], ignore_index=True)
        
        if len(result) > n_samples:
            result = result.sample(n=n_samples, random_state=self.random_seed)
        
        return result

    def _combined_sample(
        self,
        df: pd.DataFrame,
        n_samples: int,
        max_per_mention: int = 3
    ) -> pd.DataFrame:
        """Combined strategy: stratify by type, then diversify by mention."""
        if len(df) == 0:
            return df
        
        if 'entity_type' not in df.columns:
            return self._mention_diverse_sample(df, n_samples, max_per_mention)
        
        types = df['entity_type'].unique()
        per_type = max(1, n_samples // len(types))
        
        samples = []
        for etype in types:
            type_df = df[df['entity_type'] == etype]
            type_samples = self._mention_diverse_sample(type_df, per_type, max_per_mention)
            samples.append(type_samples)
        
        result = pd.concat(samples, ignore_index=True)
        
        if len(result) > n_samples:
            result = result.sample(n=n_samples, random_state=self.random_seed)
        
        return result

    def compute_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics about the sample."""
        stats = {
            'total_samples': len(df),
            'unique_mentions': df['mention'].str.lower().nunique(),
            'unique_entities': df[df['linked_to_concept_id'] != '[UNLINKED]']['linked_to_concept_id'].nunique() if 'linked_to_concept_id' in df.columns else 0,
            'source_files': df['source_file'].nunique() if 'source_file' in df.columns else 0,
            'linked_count': df['is_linked'].sum() if 'is_linked' in df.columns else len(df),
            'unlinked_count': (~df['is_linked']).sum() if 'is_linked' in df.columns else 0,
            'type_distribution': df['entity_type'].value_counts().to_dict() if 'entity_type' in df.columns else {},
            'mention_frequency': df['mention'].str.lower().value_counts().head(20).to_dict(),
            'linking_model_distribution': df['linking_model'].value_counts().to_dict() if 'linking_model' in df.columns else {},
            'avg_context_length': df['context'].str.len().mean() if 'context' in df.columns else 0
        }
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract diverse samples from NER+EL output for expert annotation"
    )
    
    # Directory-based mode
    parser.add_argument("--sections-dir", "-sd", help="Directory containing sections CSV files")
    parser.add_argument("--ner-dir", "-nd", help="Directory containing NER output JSONL files")
    
    # Single-file mode
    parser.add_argument("--sections", "-s", help="Path to single sections CSV")
    parser.add_argument("--ner-output", "-n", help="Path to single NER output JSONL")
    
    # Required
    parser.add_argument("--output", "-o", required=True, help="Output TSV path")
    
    # Optional taxonomy (no longer required)
    parser.add_argument("--taxonomy", "-t", help="Path to taxonomy TSV (optional, not used)")
    
    # Sampling parameters
    parser.add_argument("--n-files", type=int, default=50, help="Number of files to sample (0 for all)")
    parser.add_argument("--file-strategy", choices=['weighted', 'uniform', 'all', 'model_balanced'], default='weighted')
    parser.add_argument("--n-samples", type=int, default=100, help="Number of entity samples")
    parser.add_argument("--strategy", choices=['random', 'stratified', 'mention_diverse', 'combined', 'model_stratified'], default='combined')
    parser.add_argument("--context-window", type=int, default=100)
    parser.add_argument("--use-sentences", action="store_true", default=True)
    parser.add_argument("--all-samples", action="store_true", help="Output all samples")
    parser.add_argument("--stats", action="store_true", help="Print statistics")
    parser.add_argument("--seed", type=int, default=42)
    
    # Unlinked entity options
    parser.add_argument("--include-unlinked", action="store_true", help="Include unlinked entities in samples")
    parser.add_argument("--unlinked-ratio", type=float, default=0.2, help="Target ratio of unlinked entities (default: 0.2)")
    
    # Entity type filter (for cancer domain)
    parser.add_argument("--entity-type", "-et", help="Filter by NER entity type (e.g., Gene, Disease, Species)")
    
    args = parser.parse_args()
    
    # Warn if taxonomy provided (not needed anymore)
    if args.taxonomy:
        print("⚠️  Note: --taxonomy is no longer required (entity types come from NER output)")
    
    # Validate arguments
    if args.sections_dir and args.ner_dir:
        mode = 'directory'
    elif args.sections and args.ner_output:
        mode = 'single'
    else:
        parser.error("Provide either (--sections-dir + --ner-dir) or (--sections + --ner-output)")
    
    # Initialize extractor
    if mode == 'directory':
        extractor = SampleExtractor(
            sections_dir=args.sections_dir,
            ner_output_dir=args.ner_dir,
            context_window=args.context_window,
            use_sentence_context=args.use_sentences,
            random_seed=args.seed,
            include_unlinked=args.include_unlinked,
            unlinked_ratio=args.unlinked_ratio
        )
    else:
        extractor = SampleExtractor(
            sections_path=args.sections,
            ner_output_path=args.ner_output,
            context_window=args.context_window,
            use_sentence_context=args.use_sentences,
            random_seed=args.seed,
            include_unlinked=args.include_unlinked,
            unlinked_ratio=args.unlinked_ratio
        )
    
    # Sample files
    n_files = args.n_files if args.n_files > 0 else len(extractor.file_pairs)
    file_strategy = 'all' if args.n_files == 0 else args.file_strategy
    
    # For model_balanced strategy, scan files first
    model_stats_df = None
    if file_strategy == 'model_balanced':
        print(f"\n📊 Scanning files for model distribution...")
        model_stats_df, global_counts = extractor.scan_model_distribution()
        print(f"   Total entities by model:")
        for model, count in sorted(global_counts.items(), key=lambda x: -x[1]):
            print(f"      {model}: {count}")
    
    print(f"\n🗂️  Sampling {n_files} files (strategy: {file_strategy})...")
    sampled_files = extractor.sample_files(n_files, strategy=file_strategy, model_stats_df=model_stats_df)
    print(f"   Selected {len(sampled_files)} files")
    
    # Extract samples
    print("\n🔍 Extracting samples from selected files...")
    all_samples = extractor.extract_samples_from_files(sampled_files)
    print(f"   Found {len(all_samples)} total entity mentions")
    
    # Filter by entity type if specified
    if args.entity_type and len(all_samples) > 0:
        entity_type_lower = args.entity_type.lower()
        all_samples = all_samples[
            all_samples['entity_type'].str.lower() == entity_type_lower
        ]
        print(f"   Filtered to entity type '{args.entity_type}': {len(all_samples)} samples")
    
    # Split clean samples and mismatches
    if 'offset_mismatch' in all_samples.columns:
        mismatches = all_samples[all_samples['offset_mismatch'] == True].copy()
        clean_samples = all_samples[all_samples['offset_mismatch'] == False].copy()
        print(f"   Clean samples: {len(clean_samples)}, Offset mismatches: {len(mismatches)}")
    else:
        clean_samples = all_samples
        mismatches = pd.DataFrame()
    
    # Show linked/unlinked stats
    if 'is_linked' in clean_samples.columns and len(clean_samples) > 0:
        n_linked = clean_samples['is_linked'].sum()
        n_unlinked = (~clean_samples['is_linked']).sum()
        print(f"   Linked: {n_linked}, Unlinked: {n_unlinked}")
    
    # Handle empty results
    if len(clean_samples) == 0:
        print(f"\n⚠️  No samples found" + (f" for entity type '{args.entity_type}'" if args.entity_type else ""))
        return
    
    # Select samples
    if args.all_samples:
        selected = clean_samples
        print(f"   Outputting all {len(selected)} clean samples")
    else:
        print(f"\n🎯 Selecting {args.n_samples} diverse samples (strategy: {args.strategy})...")
        selected = extractor.select_diverse_samples(
            clean_samples,
            n_samples=args.n_samples,
            strategy=args.strategy
        )
        print(f"   Selected {len(selected)} samples")
    
    # Add annotation columns
    selected = selected.copy()
    selected['wrong_mention'] = ''
    selected['correct_concept_id'] = ''
    selected['notes'] = ''
    
    # Remove internal columns, reorder for annotation
    output_columns = [
        'id', 'mention', 'context', 
        'linking_model', 'linked_to_concept_id', 'linked_to_name', 
        'entity_type',
        'linked_to_wikidata_id', 'linked_to_wikidata_name',
        'wrong_mention', 'correct_concept_id', 'notes', 'source_file'
    ]
    selected = selected[[c for c in output_columns if c in selected.columns]]
    
    # Print statistics
    if args.stats:
        print("\n📈 Sample Statistics:")
        stats = extractor.compute_statistics(selected)
        print(f"   Total samples: {stats['total_samples']}")
        print(f"   Unique mentions: {stats['unique_mentions']}")
        print(f"   Unique linked entities: {stats['unique_entities']}")
        print(f"   Source files: {stats['source_files']}")
        print(f"   Linked: {stats['linked_count']}, Unlinked: {stats['unlinked_count']}")
        print(f"   Avg context length: {stats['avg_context_length']:.0f} chars")
        
        if stats['type_distribution']:
            print(f"   Entity types: {stats['type_distribution']}")
        
        if stats['linking_model_distribution']:
            print(f"   Linking models: {stats['linking_model_distribution']}")
    
    # Save output
    selected.to_csv(args.output, sep='\t', index=False)
    print(f"\n✅ Saved {len(selected)} samples to {args.output}")


if __name__ == "__main__":
    main()
