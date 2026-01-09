#!/usr/bin/env python3
"""
Fix gazetteer entities in existing NER outputs.
Updates: model name, entity type, linking source
"""

import json
import pandas as pd
from pathlib import Path
import shutil
import argparse
import sys

# Add project to path
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


def fix_gazetteer_output(file_path, taxonomy_source, model_name, default_type, domain, id_to_type):
    """Fix gazetteer entities in a single file."""
    sections = []
    fixed_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            section = json.loads(line)
            
            for entity in section.get('entities', []):
                # Only fix gazetteer entities (check for any *-Gazetteer pattern)
                if 'Gazetteer' in entity.get('model', ''):
                    # Fix model name
                    entity['model'] = model_name
                    
                    # Fix entity type
                    entity_id = entity.get('linking', [{}])[0].get('id', '')
                    taxonomy_type = id_to_type.get(entity_id, '')
                    entity['entity'] = map_type(taxonomy_type, domain, default_type)
                    
                    # Fix linking source
                    if entity.get('linking'):
                        entity['linking'][0]['source'] = taxonomy_source
                    
                    fixed_count += 1
            
            sections.append(section)
    
    return sections, fixed_count


def process_domain(domain, ner_output_dir, dry_run=False):
    """Process all files for a domain."""
    print(f"\nProcessing {domain} domain...")
    
    config = DOMAIN_MODELS.get(domain)
    if not config:
        print(f"  ERROR: Domain '{domain}' not found in DOMAIN_MODELS")
        return
    
    if not config.get('gazetteer', {}).get('enabled'):
        print(f"  Skipped: gazetteer not enabled for this domain")
        return
    
    gaz_conf = config['gazetteer']
    taxonomy_source = gaz_conf.get('taxonomy_source', 'Unknown')
    model_name = gaz_conf.get('model_name', f'{domain}-Gazetteer')
    default_type = gaz_conf.get('default_type')
    
    print(f"  Taxonomy source: {taxonomy_source}")
    print(f"  Model name: {model_name}")
    print(f"  Default type: {default_type}")
    
    # Load taxonomy
    taxonomy_path = Path(__file__).parent.parent / gaz_conf['taxonomy_path']
    if not taxonomy_path.exists():
        print(f"  ERROR: Taxonomy file not found: {taxonomy_path}")
        return
    
    id_to_type = load_taxonomy_lookup(taxonomy_path)
    print(f"  Loaded taxonomy: {len(id_to_type)} entries")
    
    # Find output files
    output_path = Path(ner_output_dir)
    if not output_path.exists():
        print(f"  ERROR: Output directory not found: {output_path}")
        return
    
    files = list(output_path.glob('*.jsonl'))
    print(f"  Found {len(files)} JSONL files")
    
    if dry_run:
        print(f"  DRY RUN MODE - No files will be modified")
    
    total_fixed = 0
    
    # Create backup directory
    backup_dir = output_path / 'BKP'
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
        print(f"  Backup directory: {backup_dir}")
    
    for i, file_path in enumerate(files, 1):
        # Backup original to BKP subdirectory
        if not dry_run:
            backup_path = backup_dir / f"{file_path.name}.backup"
            if not backup_path.exists():
                shutil.copy(file_path, backup_path)
        
        # Fix entities
        sections, fixed_count = fix_gazetteer_output(
            file_path, taxonomy_source, model_name, default_type, domain, id_to_type
        )
        total_fixed += fixed_count
        
        # Write back
        if not dry_run:
            with open(file_path, 'w', encoding='utf-8') as f:
                for section in sections:
                    f.write(json.dumps(section, ensure_ascii=False) + '\n')
        
        if fixed_count > 0:
            print(f"    [{i}/{len(files)}] {file_path.name}: {fixed_count} entities fixed")
    
    print(f"\n  TOTAL: Fixed {total_fixed} gazetteer entities across {len(files)} files")
    if dry_run:
        print(f"  (DRY RUN - no changes were made)")


def main():
    parser = argparse.ArgumentParser(
        description='Fix gazetteer entities in existing NER outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would change
  python fix_gazetteer_outputs.py --domain neuro --ner_output_dir outputs/neuro/ner --dry-run
  
  # Apply fixes
  python fix_gazetteer_outputs.py --domain neuro --ner_output_dir outputs/neuro/ner
  
  # Fix all domains
  for domain in neuro maritime energy; do
    python fix_gazetteer_outputs.py --domain $domain --ner_output_dir outputs/$domain/ner
  done
        """
    )
    parser.add_argument('--domain', required=True, 
                       help='Domain name (neuro, energy, maritime)')
    parser.add_argument('--ner_output_dir', required=True,
                       help='Path to NER output directory containing .jsonl files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without modifying files')
    
    args = parser.parse_args()
    
    process_domain(args.domain, args.ner_output_dir, args.dry_run)
    print("\nDone!")


if __name__ == '__main__':
    main()
