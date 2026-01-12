#!/usr/bin/env python3
"""
Analyze entity distribution from existing NER output.
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def analyze_ner_output(ner_dir: str):
    """Analyze entity distribution from NER JSONL files."""
    
    # Find all JSONL files
    jsonl_files = list(Path(ner_dir).glob("*.jsonl"))
    print(f"📂 Found {len(jsonl_files)} NER output files")
    
    # Statistics
    entity_counts = Counter()  # Total mentions per type
    unique_entities = defaultdict(set)  # Unique mentions per type
    linked_counts = Counter()  # How many are already linked
    unlinked_examples = defaultdict(list)  # Examples needing linking
    
    total_sections = 0
    
    for jsonl_path in jsonl_files:
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                section = json.loads(line)
                total_sections += 1
                
                for ent in section.get('entities', []):
                    ent_type = ent.get('entity', 'Unknown')
                    ent_text = ent.get('text', '')
                    has_linking = bool(ent.get('linking'))
                    model = ent.get('model', '')
                    
                    # Normalize type (AIONER outputs lowercase)
                    ent_type_normalized = ent_type.lower()
                    
                    entity_counts[ent_type_normalized] += 1
                    unique_entities[ent_type_normalized].add(ent_text.lower())
                    
                    if has_linking:
                        linked_counts[ent_type_normalized] += 1
                    else:
                        # Keep examples of unlinked entities
                        if len(unlinked_examples[ent_type_normalized]) < 30:
                            unlinked_examples[ent_type_normalized].append({
                                'text': ent_text,
                                'score': round(ent.get('score', 0), 3),
                                'model': model
                            })
    
    # Print results
    print(f"\n📊 Analyzed {total_sections} sections from {len(jsonl_files)} files")
    print("=" * 70)
    print(f"{'Entity Type':<15} {'Total':<10} {'Unique':<10} {'Linked':<10} {'Unlinked':<10} {'Link %':<10}")
    print("=" * 70)
    
    for ent_type in sorted(entity_counts.keys()):
        total = entity_counts[ent_type]
        unique = len(unique_entities[ent_type])
        linked = linked_counts[ent_type]
        unlinked = total - linked
        link_pct = (linked / total * 100) if total > 0 else 0
        
        print(f"{ent_type:<15} {total:<10} {unique:<10} {linked:<10} {unlinked:<10} {link_pct:<.1f}%")
    
    print("=" * 70)
    
    # Show unlinked examples for Gene and Species
    print("\n📝 UNLINKED ENTITY EXAMPLES (need linking):")
    
    for ent_type in ['gene', 'species']:
        if ent_type in unlinked_examples:
            print(f"\n{ent_type.upper()} (showing up to 20):")
            examples = unlinked_examples[ent_type][:20]
            unique_texts = list(set(e['text'] for e in examples))[:20]
            print(f"  {unique_texts}")
    
    # Summary for linking strategy
    print("\n" + "=" * 70)
    print("💡 LINKING REQUIREMENTS SUMMARY")
    print("=" * 70)
    
    for ent_type in ['gene', 'species', 'disease', 'cellline', 'chemical']:
        if ent_type in entity_counts:
            total = entity_counts[ent_type]
            unique = len(unique_entities[ent_type])
            linked = linked_counts[ent_type]
            unlinked_unique = len(set(
                e['text'].lower() for e in unlinked_examples.get(ent_type, [])
            ))
            
            if linked < total:
                print(f"\n{ent_type.upper()}:")
                print(f"  Total mentions needing linking: {total - linked:,}")
                print(f"  Unique strings to resolve: ~{unique - (linked > 0 and unique // 2 or 0):,}")
                
                # Estimate full corpus (assuming 3 files is small sample)
                if len(jsonl_files) < 100:
                    est_factor = 1000 / len(jsonl_files)  # Rough projection
                    print(f"  ⚠️  Projected for 1000 files: ~{int((total-linked) * est_factor):,} mentions")

    return {
        'entity_counts': dict(entity_counts),
        'unique_counts': {k: len(v) for k, v in unique_entities.items()},
        'linked_counts': dict(linked_counts),
        'unlinked_examples': {k: v[:10] for k, v in unlinked_examples.items()}
    }


if __name__ == "__main__":
    import sys
    
    ner_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/cancer-archive_1-ft/ner"
    analyze_ner_output(ner_dir)

