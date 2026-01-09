#!/usr/bin/env python3
"""
Add common synonyms to NCBI Species taxonomy.

This script patches the NCBI_SPECIES.tsv file with common synonyms
that are frequently used in biomedical literature but missing from
the original taxonomy.

Usage:
    python patch_species_synonyms.py \
        --input taxonomies/cancer/NCBI_SPECIES.tsv \
        --output taxonomies/cancer/NCBI_SPECIES.tsv

    # Or output to new file
    python patch_species_synonyms.py \
        --input taxonomies/cancer/NCBI_SPECIES.tsv \
        --output taxonomies/cancer/NCBI_SPECIES_enriched.tsv
"""

import argparse
import pandas as pd
from pathlib import Path


# =============================================================================
# Curated synonyms to add
# =============================================================================
# Format: NCBI_ID -> list of synonyms to add
# These are common names used in biomedical literature

SYNONYMS_TO_ADD = {
    # Mammals - Model organisms
    "NCBI:9606": [
        "human", "humans"
        # Note: "patient", "patients", "man", "men", "woman", "women" 
        # are debatable - uncomment if desired
        # "patient", "patients", "man", "men", "woman", "women",
        # "inpatient", "outpatient"
    ],
    "NCBI:10090": [
        "mouse", "mice", "murine"
    ],
    "NCBI:10116": [
        "rat", "rats"
    ],
    "NCBI:9615": [
        "dog", "dogs", "canine"
    ],
    "NCBI:9685": [
        "cat", "cats", "feline"
    ],
    "NCBI:9940": [
        "sheep", "ovine"
    ],
    "NCBI:9913": [
        "cow", "cows", "bovine", "cattle"
    ],
    "NCBI:9925": [
        "goat", "goats", "caprine"
    ],
    "NCBI:9823": [
        "pig", "pigs", "porcine", "swine"
    ],
    "NCBI:9986": [
        "rabbit", "rabbits"
    ],
    "NCBI:9031": [
        "chicken", "chickens"
    ],
    
    # Other model organisms
    "NCBI:7955": [
        "zebrafish"
    ],
    "NCBI:7227": [
        "fruit fly", "fruitfly"
    ],
    "NCBI:6239": [
        "nematode", "roundworm"
    ],
    "NCBI:4932": [
        "yeast", "baker's yeast", "budding yeast"
    ],
    
    # Common bacteria
    "NCBI:562": [
        "e. coli", "e.coli"
    ],
    "NCBI:210": [
        "h. pylori", "h.pylori"
    ],
    "NCBI:1280": [
        "s. aureus", "s.aureus", "staph aureus", "staph"
    ],
    "NCBI:1313": [
        "s. pneumoniae", "pneumococcus"
    ],
    "NCBI:287": [
        "p. aeruginosa", "pseudomonas"
    ],
    "NCBI:1351": [
        "e. faecalis", "enterococcus"
    ],
    "NCBI:1639": [
        "l. monocytogenes", "listeria"
    ],
    "NCBI:727": [
        "h. influenzae", "haemophilus"
    ],
    "NCBI:485": [
        "n. gonorrhoeae", "gonococcus"
    ],
    "NCBI:1496": [
        "c. difficile", "c. diff", "c.diff"
    ],
    "NCBI:1428": [
        "b. subtilis", "bacillus"
    ],
    "NCBI:90371": [
        "salmonella", "s. enterica"
    ],
    "NCBI:573": [
        "klebsiella", "k. pneumoniae"
    ],
    "NCBI:1763": [
        "mycobacterium", "m. tuberculosis", "mtb", "tb"
    ],
    
    # Common viruses
    "NCBI:11676": [
        "hiv", "hiv-1", "human immunodeficiency virus", 
        "human immunodeficiency virus-1", "aids virus"
    ],
    "NCBI:11709": [
        "hiv-2", "human immunodeficiency virus-2"
    ],
    "NCBI:10376": [
        "ebv", "epstein-barr virus", "epstein barr virus"
    ],
    "NCBI:10359": [
        "cmv", "hcmv", "cytomegalovirus", "human cytomegalovirus"
    ],
    "NCBI:10298": [
        "hsv", "hsv-1", "herpes simplex virus", "herpes"
    ],
    "NCBI:10310": [
        "hsv-2", "herpes simplex virus 2"
    ],
    "NCBI:10566": [
        "hpv", "human papillomavirus", "papillomavirus"
    ],
    "NCBI:333761": [
        "hpv16", "hpv-16", "human papillomavirus 16"
    ],
    "NCBI:333760": [
        "hpv18", "hpv-18", "human papillomavirus 18"
    ],
    "NCBI:10407": [
        "hbv", "hepatitis b virus", "hepatitis b"
    ],
    "NCBI:3052230": [
        "hcv", "hepatitis c virus", "hepatitis c"
    ],
    "NCBI:11320": [
        "influenza", "flu", "influenza a"
    ],
    "NCBI:2697049": [
        "sars-cov-2", "covid-19", "covid", "coronavirus"
    ],
    "NCBI:37124": [
        "kshv", "kaposi sarcoma herpesvirus", "hhv-8", "hhv8"
    ],
    "NCBI:10335": [
        "vzv", "varicella zoster virus", "chickenpox virus"
    ],
    "NCBI:10245": [
        "vaccinia", "vaccinia virus"
    ],
    "NCBI:12637": [
        "dengue", "dengue virus"
    ],
    "NCBI:64320": [
        "zika", "zika virus"
    ],
    "NCBI:11234": [
        "measles", "measles virus"
    ],
    "NCBI:37296": [
        "hdv", "hepatitis d virus", "hepatitis delta virus"
    ],
    "NCBI:1570291": [
        "bkv", "bk virus", "bk polyomavirus"
    ],
    "NCBI:1891767": [
        "sads-cov", "swine acute diarrhea syndrome coronavirus"
    ],
    
    # Fungi
    "NCBI:5476": [
        "c. albicans", "candida"
    ],
    "NCBI:5507": [
        "c. neoformans", "cryptococcus"
    ],
    "NCBI:5062": [
        "a. fumigatus", "aspergillus"
    ],
    
    # Other animals mentioned in your data
    "NCBI:9627": [
        "donkey", "donkeys"
    ],
    "NCBI:9305": [
        "tasmanian devil", "tasmanian devils", "devil", "devils"
    ],
}


def patch_synonyms(
    input_path: str,
    output_path: str,
    synonyms_map: dict = SYNONYMS_TO_ADD,
    backup: bool = True
):
    """
    Patch species TSV with additional synonyms.
    
    Args:
        input_path: Path to input TSV
        output_path: Path to output TSV (can be same as input)
        synonyms_map: Dict mapping NCBI ID to list of synonyms to add
        backup: Create backup if overwriting
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
    
    for ncbi_id, new_synonyms in synonyms_map.items():
        # Find row with this ID
        mask = df['id'] == ncbi_id
        
        if not mask.any():
            not_found.append(ncbi_id)
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
            print(f"   ✅ {ncbi_id} ({concept}): added {len(synonyms_to_add)} synonyms")
    
    # Report not found
    if not_found:
        print(f"\n⚠️  {len(not_found)} IDs not found in taxonomy:")
        for ncbi_id in not_found:
            print(f"      {ncbi_id}")
    
    # Save
    print(f"\n💾 Saving to {output_path}")
    df.to_csv(output_path, sep='\t', index=False)
    
    print(f"\n✅ Done! Updated {updated_count} entries")
    
    return updated_count, not_found


def main():
    parser = argparse.ArgumentParser(
        description="Add common synonyms to NCBI Species taxonomy"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input NCBI_SPECIES.tsv"
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
    
    args = parser.parse_args()
    
    if args.list_synonyms:
        print("Synonyms to be added:\n")
        for ncbi_id, synonyms in sorted(SYNONYMS_TO_ADD.items()):
            print(f"{ncbi_id}: {', '.join(synonyms)}")
        return
    
    patch_synonyms(
        input_path=args.input,
        output_path=args.output,
        backup=not args.no_backup
    )


if __name__ == "__main__":
    main()
