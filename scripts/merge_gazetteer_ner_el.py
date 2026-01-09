#!/usr/bin/env python3
"""
Merge gazetteer entities with existing NER entities.
Replaces old gazetteer entities with new ones, preserves NER entities.
"""

import json
from pathlib import Path
import shutil
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.domain_models import DOMAIN_MODELS


def merge_entities(old_section, new_gaz_section):
    """
    Merge entities from two sections:
    - Keep all NER entities from old_section
    - Replace gazetteer entities with ones from new_gaz_section
    """
    # Extract NER entities (non-gazetteer)
    ner_entities = [
        e for e in old_section.get('entities', [])
        if 'Gazetteer' not in e.get('model', '')
    ]
    
    # Get new gazetteer entities
    gaz_entities = new_gaz_section.get('entities', [])
    
    # Combine
    merged_entities = gaz_entities + ner_entities
    
    # Sort by start position
    merged_entities.sort(key=lambda e: e.get('start', 0))
    
    return merged_entities


def merge_files(old_ner_file, new_gaz_file):
    """Merge entities from two files section by section."""
    # Load both files
    old_sections = {}
    with open(old_ner_file, 'r', encoding='utf-8') as f:
        for line in f:
            section = json.loads(line)
            old_sections[section['section_id']] = section
    
    new_gaz_sections = {}
    with open(new_gaz_file, 'r', encoding='utf-8') as f:
        for line in f:
            section = json.loads(line)
            new_gaz_sections[section['section_id']] = section
    
    # Merge
    merged_sections = []
    for section_id, old_section in old_sections.items():
        new_gaz_section = new_gaz_sections.get(section_id, {'entities': []})
        
        merged_entities = merge_entities(old_section, new_gaz_section)
        
        merged_section = old_section.copy()
        merged_section['entities'] = merged_entities
        merged_sections.append(merged_section)
    
    return merged_sections


def process_domain(domain, old_ner_dir, new_gaz_dir, output_dir, dry_run=False, create_bkp=False):
    """Process all files for a domain."""
    print(f"\nMerging {domain} domain...")
    
    old_ner_path = Path(old_ner_dir)
    new_gaz_path = Path(new_gaz_dir)
    output_path = Path(output_dir)
    
    if not old_ner_path.exists():
        print(f"  ERROR: Old NER dir not found: {old_ner_path}")
        return
    
    if not new_gaz_path.exists():
        print(f"  ERROR: New gazetteer dir not found: {new_gaz_path}")
        return
    
    # Get files
    old_files = {f.name: f for f in old_ner_path.glob('*.jsonl')}
    new_files = {f.name: f for f in new_gaz_path.glob('*.jsonl')}
    
    common_files = set(old_files.keys()) & set(new_files.keys())
    print(f"  Found {len(common_files)} files to merge")
    
    if dry_run:
        print(f"  DRY RUN - No output")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create backup of old NER
    if create_bkp:
        backup_dir = output_path / 'BKP_OLD_NER'
        backup_dir.mkdir(exist_ok=True)
    
    total_gaz = 0
    total_ner = 0
    
    for i, filename in enumerate(sorted(common_files), 1):
        # Merge
        merged_sections = merge_files(old_files[filename], new_files[filename])
        
        # Count
        gaz_count = sum(
            1 for s in merged_sections 
            for e in s['entities'] 
            if 'Gazetteer' in e.get('model', '')
        )
        ner_count = sum(
            1 for s in merged_sections 
            for e in s['entities'] 
            if 'Gazetteer' not in e.get('model', '')
        )
        
        total_gaz += gaz_count
        total_ner += ner_count
        
        # Backup old
        if create_bkp:
            shutil.copy(old_files[filename], backup_dir / filename)
        
        # Write merged
        output_file = output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            for section in merged_sections:
                f.write(json.dumps(section, ensure_ascii=False) + '\n')
        
        print(f"    [{i}/{len(common_files)}] {filename}: {gaz_count} gaz + {ner_count} NER")
    
    print(f"\n  TOTAL: {total_gaz} gazetteer + {total_ner} NER entities")
    print(f"  Output: {output_path}")
    if create_bkp:
        print(f"  Backup: {backup_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Merge corrected gazetteer with existing NER output'
    )
    parser.add_argument('--domain', required=True)
    parser.add_argument('--old-ner-dir', required=True,
                       help='Directory with old NER output (has wrong gazetteer)')
    parser.add_argument('--new-gaz-dir', required=True,
                       help='Directory with new gazetteer output (corrected)')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for merged files')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--create-bkp', action='store_true')
    
    args = parser.parse_args()
    
    process_domain(
        args.domain,
        args.old_ner_dir,
        args.new_gaz_dir,
        args.output_dir,
        args.dry_run,
        args.create_bkp
    )
    print("\nDone!")


if __name__ == '__main__':
    main()
