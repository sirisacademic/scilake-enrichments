#!/usr/bin/env python3
"""
Add common synonyms to DOID Disease taxonomy.

This script patches the DOID_DISEASE.tsv file with common synonyms
that are frequently used in biomedical/cancer literature but missing from
the original taxonomy.

Usage:
    python patch_disease_synonyms.py \
        --input taxonomies/cancer/DOID_DISEASE.tsv \
        --output taxonomies/cancer/DOID_DISEASE.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


# =============================================================================
# Curated synonyms to add
# =============================================================================
# Format: DOID_ID -> list of synonyms to add

SYNONYMS_TO_ADD = {
    # === Common cancers with alternate names ===
    
    # Cancer / tumor general terms
    "DOID:162": [
        "cancers", "malignancy", "malignancies"
    ],
    "DOID:14566": [  # disease of cellular proliferation -> tumor/neoplasm
        "tumor", "tumors", "tumour", "tumours", "neoplasm", "neoplasms"
    ],
    
    # Stomach/Gastric cancer
    "DOID:10534": [
        "gastric cancer", "gastric carcinoma", "stomach carcinoma"
    ],
    
    # Bladder cancer  
    "DOID:11054": [
        "bladder cancer", "bladder carcinoma"
    ],
    
    # Lung cancers
    "DOID:3908": [  # lung non-small cell carcinoma
        "non-small cell lung cancer", "nsclc", "non-small-cell lung cancer"
    ],
    "DOID:1324": [  # lung cancer
        "lung carcinoma"
    ],
    
    # Breast cancer subtypes
    "DOID:0060081": [  # triple-receptor negative breast cancer
        "triple-negative breast cancer", "tnbc", "triple negative breast cancer"
    ],
    
    # Liver cancer
    "DOID:684": [  # hepatocellular carcinoma
        "hcc", "liver cancer", "hepatoma"
    ],
    
    # Colorectal cancer
    "DOID:9256": [  # colorectal cancer
        "crc", "bowel cancer"
    ],
    
    # Brain tumors
    "DOID:3068": [  # glioblastoma
        "gbm", "glioblastoma multiforme"
    ],
    "DOID:0060108": [  # brain glioma
        "glioma", "gliomas"
    ],
    
    # Thyroid cancer
    "DOID:0080522": [  # anaplastic thyroid carcinoma
        "anaplastic thyroid cancer", "atc"
    ],
    
    # Esophageal cancer
    "DOID:3748": [  # esophagus squamous cell carcinoma
        "esophageal squamous cell carcinoma", "escc", 
        "oesophageal squamous cell carcinoma"
    ],
    
    # Lymphomas
    "DOID:0080092": [  # primary effusion lymphoma (if exists)
        "pel"
    ],
    
    # Leukemias
    "DOID:9119": [  # acute myeloid leukemia
        "aml"
    ],
    "DOID:9952": [  # acute lymphoblastic leukemia
        "all"
    ],
    
    # Other cancers
    "DOID:1909": [  # melanoma
        "malignant melanoma"
    ],
    "DOID:9538": [  # multiple myeloma
        "mm"
    ],
    
    # === Common conditions (non-cancer) ===
    
    # Neurological
    "DOID:10652": [  # Alzheimer's disease
        "ad", "alzheimer", "alzheimers"
    ],
    "DOID:1596": [  # depressive disorder
        "depression"
    ],
    "DOID:2030": [  # anxiety disorder
        "anxiety"
    ],
    
    # Cardiovascular
    "DOID:6000": [  # congestive heart failure
        "heart failure", "chf", "cardiac failure"
    ],
    
    # Metabolic
    "DOID:9351": [  # diabetes mellitus
        "diabetes"
    ],
    "DOID:0090109": [  # autosomal dominant hypocalcemia
        "hypocalcemia"
    ],
    
    # Inflammatory conditions
    "DOID:0050589": [  # inflammatory bowel disease
        "ibd"
    ],
    "DOID:0060180": [  # colitis
        "ulcerative colitis", "uc"
    ],
    
    # === Generic/process terms - map to closest disease concept ===
    # These are borderline - they're biological processes, not diseases
    # But in cancer literature they often refer to disease states
    
    "DOID:9953": [  # hypoxia (if exists as disease)
        # Skip - hypoxia is a condition, not a disease
    ],
    
    # Metastasis - map to "metastatic neoplasm" or similar
    "DOID:169": [  # neoplasm metastasis (secondary malignant neoplasm)
        "metastasis", "metastases", "metastatic"
    ],
}

# Terms to skip (not diseases, too generic, or false positives)
SKIP_TERMS = {
    "inflammatory",  # adjective, not disease
    "inflammation",  # process, not disease
    "toxicity",      # adverse effect, not disease
    "death",         # outcome, not disease
    "pain",          # symptom, not disease
    "hypoxia",       # condition, not disease
    "infection",     # too generic
    "ferroptosis",   # cell death mechanism, not disease
    "lvi",           # lymphovascular invasion - pathological feature, not disease
}


def patch_synonyms(
    input_path: str,
    output_path: str,
    synonyms_map: dict = SYNONYMS_TO_ADD,
    backup: bool = True
):
    """
    Patch disease TSV with additional synonyms.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    print(f"📂 Reading {input_path}")
    df = pd.read_csv(input_path, sep='\t', dtype=str)
    df = df.fillna('')
    
    print(f"   Total entries: {len(df):,}")
    
    # Create backup if overwriting
    if backup and input_path == output_path:
        backup_path = input_path.with_suffix('.tsv.bak')
        df.to_csv(backup_path, sep='\t', index=False)
        print(f"   Backup created: {backup_path}")
    
    # Track changes
    updated_count = 0
    not_found = []
    
    for doid_id, new_synonyms in synonyms_map.items():
        if not new_synonyms:  # Skip empty lists
            continue
            
        # Find row with this ID
        mask = df['id'] == doid_id
        
        if not mask.any():
            not_found.append(doid_id)
            continue
        
        # Get current synonyms
        idx = df[mask].index[0]
        current_synonyms = df.loc[idx, 'synonyms']
        
        # Parse existing synonyms into set (case-insensitive check)
        if current_synonyms:
            existing = set(s.strip().lower() for s in current_synonyms.split('|'))
        else:
            existing = set()
        
        # Add new synonyms that don't already exist
        synonyms_to_add = []
        for syn in new_synonyms:
            if syn.lower() not in existing:
                synonyms_to_add.append(syn)
        
        if synonyms_to_add:
            # Combine with existing
            if current_synonyms:
                updated_synonyms = current_synonyms + '|' + '|'.join(synonyms_to_add)
            else:
                updated_synonyms = '|'.join(synonyms_to_add)
            
            df.loc[idx, 'synonyms'] = updated_synonyms
            updated_count += 1
            
            concept = df.loc[idx, 'concept']
            print(f"   ✅ {doid_id} ({concept}): added {len(synonyms_to_add)} synonyms")
    
    # Report not found
    if not_found:
        print(f"\n⚠️  {len(not_found)} IDs not found in taxonomy:")
        for doid_id in not_found:
            print(f"      {doid_id}")
    
    # Save
    print(f"\n💾 Saving to {output_path}")
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"\n✅ Done! Updated {updated_count} entries")
    
    print(f"\n📝 Note: The following terms should be SKIPPED (not diseases):")
    for term in sorted(SKIP_TERMS):
        print(f"      {term}")
    
    return updated_count, not_found


def main():
    parser = argparse.ArgumentParser(
        description="Add common synonyms to DOID Disease taxonomy"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input DOID_DISEASE.tsv"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output TSV (can be same as input)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Don't create backup when overwriting"
    )
    parser.add_argument(
        "--list-synonyms",
        action="store_true",
        help="Just list synonyms that would be added, don't modify"
    )
    parser.add_argument(
        "--list-skip",
        action="store_true",
        help="List terms that should be skipped/blocked"
    )
    
    args = parser.parse_args()
    
    if args.list_skip:
        print("Terms to SKIP (add to blocked_mentions):\n")
        for term in sorted(SKIP_TERMS):
            print(f"  {term}")
        return
    
    if args.list_synonyms:
        print("Synonyms to be added:\n")
        for doid_id, synonyms in sorted(SYNONYMS_TO_ADD.items()):
            if synonyms:
                print(f"{doid_id}: {', '.join(synonyms)}")
        return
    
    patch_synonyms(
        input_path=args.input,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
