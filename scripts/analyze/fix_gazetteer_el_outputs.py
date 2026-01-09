#!/usr/bin/env python3
"""
Fix gazetteer entities in Entity Linking outputs.
Only updates model name, entity type, and linking source for gazetteer entities.
"""

import json
import pandas as pd
from pathlib import Path
import shutil
import argparse
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.domain_models import DOMAIN_MODELS


def load_taxonomy_lookup(taxonomy_path):
    """Load taxonomy ID to type mapping."""
    df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
    return dict(zip(df['id'].astype(str), df['type']))


def map_type(taxonomy_type, domain, default_type):
    """Apply same logic as gazetteer_linker._map_type"""
    if not taxonomy_type or taxonomy_type.strip() == "":
        return default_type or 'Unknown'
    
    if domain == 'energy':
        return {'Renewables': 'energytype', 
                'Non-renewable': 'energytype', 
                'Storage': 'energystorage'}.get(taxonomy_type, 'energytype')
    elif domain == 'maritime':
        return 'vesselType'
    else:
        return taxonomy_type


def fix_gazetteer_entity(entity, taxonomy_source, model_name, default_type, domain, id_to_type):
    """Fix a single gazetteer entity."""
    if 'Gazetteer' not in entity.get('model', ''):
        return False
    
    # Fix model name
    entity['model'] = model_name
    
    # Fix entity type
    entity_id = entity.get('linking', [{}])[0].get('id', '')
    taxonomy_type = id_to_type.get(entity_id, '')
    entity['entity'] = map_type(taxonomy_type, domain, default_type)
    
    # Fix linking source (update first entry, preserve rest)
    if entity.get('linking'):
        for link in entity['linking']:
            # Only update entries that have wrong source (IRENA for non-energy domains)
            if link.get('source') in ['IRENA', 'Unknown'] or not link.get('source'):
                link['source'] = taxonomy_source
                break
    
    return True


def process_file(file_path, taxonomy_source, model_name, default_type, domain, id_to_type):
    """Process a single EL output file."""
    sections = []
    fixed_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            section = json.loads(line)
            
            for entity in section.get('entities', []):
                if fix_gazetteer_entity(entity, taxonomy_source, model_name, default_type, domain, id_to_type):
                    fixed_count += 1
            
            sections.append(section)
    
    return sections, fixed_count


def process_domain(domain, el_output_dir, dry_run=False):
    """Process all EL output files for a domain."""
    print(f"\nProcessing {domain} domain (EL outputs)...")
    
    config = DOMAIN_MODELS.get(domain)
    if not config or not config.get('gazetteer', {}).get('enabled'):
        print(f"  Skipped: gazetteer not enabled")
        return
    
    gaz_conf = config['gazetteer']
    taxonomy_source = gaz_conf.get('taxonomy_source', 'Unknown')
    model_name = gaz_conf.get('model_name', f'{domain}-Gazetteer')
    default_type = gaz_conf.get('default_type')
    
    print(f"  Taxonomy source: {taxonomy_source}")
    print(f"  Model name: {model_name}")
    
    # Load taxonomy
    taxonomy_path = Path(__file__).parent.parent / gaz_conf['taxonomy_path']
    if not taxonomy_path.exists():
        print(f"  ERROR: Taxonomy not found: {taxonomy_path}")
        return
    
    id_to_type = load_taxonomy_lookup(taxonomy_path)
    
    # Find files
    output_path = Path(el_output_dir)
    if not output_path.exists():
        print(f"  ERROR: Directory not found: {output_path}")
        return
    
    files = list(output_path.glob('*.jsonl'))
    print(f"  Found {len(files)} files")
    
    if dry_run:
        print(f"  DRY RUN - No changes")
    
    # Create backup directory
    backup_dir = output_path / 'BKP'
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
    
    total_fixed = 0
    
    for i, file_path in enumerate(files, 1):
        # Backup
        if not dry_run:
            backup_path = backup_dir / f"{file_path.name}.backup"
            if not backup_path.exists():
                shutil.copy(file_path, backup_path)
        
        # Process
        sections, fixed_count = process_file(
            file_path, taxonomy_source, model_name, default_type, domain, id_to_type
        )
        total_fixed += fixed_count
        
        # Write
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                for section in sections:
                    f.write(json.dumps(section, ensure_ascii=False) + '\n')
        
        if fixed_count > 0:
            print(f"    [{i}/{len(files)}] {file_path.name}: {fixed_count} fixed")
    
    print(f"\n  TOTAL: {total_fixed} gazetteer entities fixed")
    if dry_run:
        print(f"  (DRY RUN)")


def main():
    parser = argparse.ArgumentParser(
        description='Fix gazetteer entities in EL outputs'
    )
    parser.add_argument('--domain', required=True)
    parser.add_argument('--el_output_dir', required=True,
                       help='Path to EL output directory')
    parser.add_argument('--dry-run', action='store_true')
    
    args = parser.parse_args()
    process_domain(args.domain, args.el_output_dir, args.dry_run)
    print("\nDone!")


if __name__ == '__main__':
    main()
