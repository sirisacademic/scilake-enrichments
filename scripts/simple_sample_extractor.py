#!/usr/bin/env python3
"""
Simple sequential sample extractor for annotation.
Scans files one by one and collects unique samples until target is reached.
"""

import json
import pandas as pd
import spacy
import argparse
import os
from pathlib import Path
from typing import Dict, Set, List, Optional
from tqdm import tqdm
import random


def extract_context(nlp, text: str, start: int, end: int, context_window: int = 100) -> str:
    """Extract sentence containing the entity, or fallback to character window."""
    # Try sentence-based context
    doc = nlp(text)
    for sent in doc.sents:
        if sent.start_char <= start and end <= sent.end_char:
            return sent.text.strip()
    
    # Fallback to character window
    ctx_start = max(0, start - context_window)
    ctx_end = min(len(text), end + context_window)
    return text[ctx_start:ctx_end].strip()


def main():
    parser = argparse.ArgumentParser(description="Simple sequential sample extractor")
    parser.add_argument("--sections-dir", "-sd", required=True, help="Directory with sections CSVs")
    parser.add_argument("--ner-dir", "-nd", required=True, help="Directory with NER JSONL files")
    parser.add_argument("--output", "-o", required=True, help="Output TSV path")
    parser.add_argument("--entity-type", "-et", required=True, help="Entity type to extract (e.g., CellLine)")
    parser.add_argument("--n-linked", type=int, default=240, help="Number of linked samples")
    parser.add_argument("--n-unlinked", type=int, default=60, help="Number of unlinked samples")
    parser.add_argument("--max-per-mention", type=int, default=2, help="Max samples per unique mention text")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle file order")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    # Load spaCy
    print("📦 Loading spaCy...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    
    # Find all file pairs
    sections_dir = Path(args.sections_dir)
    ner_dir = Path(args.ner_dir)
    
    # Index sections files
    sections_index = {}
    for f in sections_dir.glob("*_sections.csv"):
        doc_id = f.stem.replace("_sections", "")
        sections_index[doc_id] = f
    
    # Get NER files that have matching sections
    ner_files = []
    for f in ner_dir.glob("*.jsonl"):
        doc_id = f.stem
        if doc_id in sections_index:
            ner_files.append({
                'doc_id': doc_id,
                'ner_path': f,
                'sections_path': sections_index[doc_id]
            })
    
    print(f"📂 Found {len(ner_files)} file pairs")
    
    if args.shuffle:
        random.shuffle(ner_files)
    
    # Tracking
    linked_samples: List[Dict] = []
    unlinked_samples: List[Dict] = []
    seen_ids: Set[str] = set()
    mention_counts: Dict[str, int] = {}  # mention_lower -> count
    
    entity_type_lower = args.entity_type.lower()
    
    # Process files sequentially
    pbar = tqdm(ner_files, desc="Scanning files")
    for fp in pbar:
        # Check if we have enough
        if len(linked_samples) >= args.n_linked and len(unlinked_samples) >= args.n_unlinked:
            print(f"\n✅ Reached target: {len(linked_samples)} linked, {len(unlinked_samples)} unlinked")
            break
        
        pbar.set_postfix({
            'linked': len(linked_samples), 
            'unlinked': len(unlinked_samples)
        })
        
        # Load sections
        try:
            sections_df = pd.read_csv(fp['sections_path'], dtype=str, engine='python')
            text_col = 'section_content_expanded' if 'section_content_expanded' in sections_df.columns else 'section_content'
            sections_map = dict(zip(
                sections_df['section_id'].fillna(''),
                sections_df[text_col].fillna('')
            ))
        except Exception as e:
            continue
        
        # Load NER output
        try:
            with open(fp['ner_path'], 'r') as f:
                ner_records = [json.loads(line) for line in f if line.strip()]
        except Exception:
            continue
        
        # Process entities
        for record in ner_records:
            section_id = record.get('section_id', '')
            section_text = sections_map.get(section_id, '')
            if not section_text:
                continue
            
            for entity in record.get('entities', []):
                # Filter by entity type
                ent_type = entity.get('entity', '').lower()
                if ent_type != entity_type_lower:
                    continue
                
                mention = entity.get('text', '')
                start = entity.get('start', 0)
                end = entity.get('end', 0)
                
                # Generate unique ID
                sample_id = f"{section_id}#{start}-{end}"
                
                # Skip duplicates
                if sample_id in seen_ids:
                    continue
                
                # Validate offset
                if start < 0 or end > len(section_text):
                    continue
                mention_from_text = section_text[start:end]
                if mention.lower() != mention_from_text.lower():
                    continue
                
                # Check mention frequency limit
                mention_lower = mention.lower()
                current_count = mention_counts.get(mention_lower, 0)
                if current_count >= args.max_per_mention:
                    continue
                
                # Check if linked
                linking = entity.get('linking')
                is_linked = linking and isinstance(linking, list) and len(linking) > 0
                
                # Check if we need more of this type
                if is_linked and len(linked_samples) >= args.n_linked:
                    continue
                if not is_linked and len(unlinked_samples) >= args.n_unlinked:
                    continue
                
                # Extract context
                context = extract_context(nlp, section_text, start, end)
                
                # Build sample
                if is_linked:
                    primary_link = linking[0]
                    linked_id = primary_link.get('id', '')
                    linked_name = primary_link.get('name', '')
                    wikidata_link = linking[1] if len(linking) > 1 and linking[1].get('source') == 'Wikidata' else {}
                    wikidata_id = wikidata_link.get('id', '')
                    wikidata_name = wikidata_link.get('name', '')
                else:
                    linked_id = '[UNLINKED]'
                    linked_name = ''
                    wikidata_id = ''
                    wikidata_name = ''
                
                sample = {
                    'id': sample_id,
                    'mention': mention,
                    'context': context,
                    'linking_model': entity.get('model', 'unknown'),
                    'linked_to_concept_id': linked_id,
                    'linked_to_name': linked_name,
                    'entity_type': entity.get('entity', ''),
                    'linked_to_wikidata_id': wikidata_id,
                    'linked_to_wikidata_name': wikidata_name,
                    'wrong_mention': '',
                    'correct_concept_id': '',
                    'notes': '',
                    'source_file': fp['doc_id']
                }
                
                # Add to appropriate list
                if is_linked:
                    linked_samples.append(sample)
                else:
                    unlinked_samples.append(sample)
                
                seen_ids.add(sample_id)
                mention_counts[mention_lower] = current_count + 1
    
    # Combine and shuffle
    all_samples = linked_samples + unlinked_samples
    random.shuffle(all_samples)
    
    # Save
    df = pd.DataFrame(all_samples)
    df.to_csv(args.output, sep='\t', index=False)
    
    # Stats
    print(f"\n📊 Results:")
    print(f"   Linked: {len(linked_samples)}")
    print(f"   Unlinked: {len(unlinked_samples)}")
    print(f"   Total: {len(all_samples)}")
    print(f"   Unique mentions: {len(mention_counts)}")
    print(f"\n✅ Saved to {args.output}")


if __name__ == "__main__":
    main()
