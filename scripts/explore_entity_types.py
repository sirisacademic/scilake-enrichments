#!/usr/bin/env python3
"""
Explore and compare entity types between NER annotation samples and taxonomy files.

This script helps identify discrepancies in type naming conventions between:
- entity_type column in annotation samples (from NER)
- type column in taxonomy files

Usage:
    python explore_entity_types.py --taxonomies-dir /path/to/taxonomies --samples-dir /path/to/samples
"""

import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import sys


def find_sample_files(samples_dir: Path) -> dict:
    """
    Find annotation sample TSV files, organized by domain.
    Returns: {domain: [list of sample files]}
    """
    samples = defaultdict(list)
    
    for tsv_file in samples_dir.rglob("*_annotation_samples*.tsv"):
        # Skip enriched files
        if "_enriched" in tsv_file.name:
            continue
        
        # Extract domain from path or filename
        # e.g., /outputs/neuro-all-ft/neuro_el_annotation_samples.tsv -> neuro
        # e.g., /outputs/cancer-all-ft/cancer_el_annotation_samples_GENE.tsv -> cancer
        parent_name = tsv_file.parent.name
        if "-all-ft" in parent_name:
            domain = parent_name.replace("-all-ft", "")
        else:
            # Try to extract from filename
            domain = tsv_file.name.split("_")[0]
        
        samples[domain].append(tsv_file)
    
    return dict(samples)


def find_taxonomy_files(taxonomies_dir: Path) -> dict:
    """
    Find taxonomy TSV files, organized by domain.
    Returns: {domain: [list of taxonomy files]}
    """
    taxonomies = defaultdict(list)
    
    for domain_dir in taxonomies_dir.iterdir():
        if domain_dir.is_dir():
            domain = domain_dir.name
            for tsv_file in domain_dir.glob("*.tsv"):
                taxonomies[domain].append(tsv_file)
    
    return dict(taxonomies)


def get_ner_types(sample_file: Path) -> dict:
    """
    Extract entity_type values and their counts from a sample file.
    Returns: {type: count}
    """
    try:
        df = pd.read_csv(sample_file, sep='\t', low_memory=False)
        if 'entity_type' in df.columns:
            return df['entity_type'].value_counts().to_dict()
    except Exception as e:
        print(f"  Warning: Could not read {sample_file.name}: {e}")
    return {}


def get_taxonomy_types(taxonomy_file: Path) -> dict:
    """
    Extract type values and their counts from a taxonomy file.
    Returns: {type: count}
    """
    try:
        df = pd.read_csv(taxonomy_file, sep='\t', low_memory=False)
        if 'type' in df.columns:
            return df['type'].value_counts().to_dict()
    except Exception as e:
        print(f"  Warning: Could not read {taxonomy_file.name}: {e}")
    return {}


def normalize_type(t: str) -> str:
    """Normalize type string for comparison (lowercase)."""
    if pd.isna(t):
        return ""
    return str(t).lower().strip()


def print_comparison(domain: str, ner_types: dict, taxonomy_types: dict):
    """Print a side-by-side comparison of types."""
    print(f"\n{'='*70}")
    print(f"DOMAIN: {domain.upper()}")
    print(f"{'='*70}")
    
    # Normalize for comparison
    ner_normalized = {normalize_type(k): k for k in ner_types.keys()}
    tax_normalized = {normalize_type(k): k for k in taxonomy_types.keys()}
    
    all_normalized = set(ner_normalized.keys()) | set(tax_normalized.keys())
    all_normalized.discard("")  # Remove empty
    
    print(f"\n{'NER Entity Type':<35} {'Taxonomy Type':<35}")
    print(f"{'(from samples)':<35} {'(from taxonomy files)':<35}")
    print("-" * 70)
    
    # Matched types (same when normalized)
    matched = []
    ner_only = []
    tax_only = []
    
    for norm_type in sorted(all_normalized):
        ner_orig = ner_normalized.get(norm_type)
        tax_orig = tax_normalized.get(norm_type)
        
        if ner_orig and tax_orig:
            ner_count = ner_types.get(ner_orig, 0)
            tax_count = taxonomy_types.get(tax_orig, 0)
            matched.append((ner_orig, ner_count, tax_orig, tax_count))
        elif ner_orig:
            ner_count = ner_types.get(ner_orig, 0)
            ner_only.append((ner_orig, ner_count))
        else:
            tax_count = taxonomy_types.get(tax_orig, 0)
            tax_only.append((tax_orig, tax_count))
    
    if matched:
        print("\n✓ MATCHED TYPES (same when case-normalized):")
        for ner_orig, ner_count, tax_orig, tax_count in sorted(matched, key=lambda x: -x[1]):
            match_marker = "=" if ner_orig == tax_orig else "≈"
            print(f"  {ner_orig:<30} ({ner_count:>4}) {match_marker} {tax_orig:<30} ({tax_count:>6})")
    
    if ner_only:
        print("\n✗ NER ONLY (not in taxonomy):")
        for orig, count in sorted(ner_only, key=lambda x: -x[1]):
            print(f"  {orig:<30} ({count:>4})")
    
    if tax_only:
        print("\n✗ TAXONOMY ONLY (not in NER samples):")
        for orig, count in sorted(tax_only, key=lambda x: -x[1]):
            print(f"  {orig:<30} ({count:>6})")
    
    # Summary
    print(f"\nSummary: {len(matched)} matched, {len(ner_only)} NER-only, {len(tax_only)} taxonomy-only")
    
    # Check for case mismatches
    case_mismatches = [(n, t) for n, _, t, _ in matched if n != t]
    if case_mismatches:
        print(f"\n⚠ CASE MISMATCHES (may need normalization):")
        for ner_orig, tax_orig in case_mismatches:
            print(f"  '{ner_orig}' vs '{tax_orig}'")


def main():
    parser = argparse.ArgumentParser(
        description='Explore and compare entity types between NER samples and taxonomies'
    )
    parser.add_argument('--taxonomies-dir', required=True,
                        help='Root directory containing taxonomy subdirectories')
    parser.add_argument('--samples-dir', required=True,
                        help='Root directory containing annotation sample files')
    parser.add_argument('--domain', default=None,
                        help='Specific domain to analyze (optional, analyzes all if not specified)')
    
    args = parser.parse_args()
    
    taxonomies_dir = Path(args.taxonomies_dir)
    samples_dir = Path(args.samples_dir)
    
    if not taxonomies_dir.exists():
        print(f"Error: Taxonomies directory not found: {taxonomies_dir}")
        sys.exit(1)
    
    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        sys.exit(1)
    
    # Find all files
    print("Scanning for taxonomy files...")
    taxonomy_files = find_taxonomy_files(taxonomies_dir)
    print(f"Found taxonomies for domains: {list(taxonomy_files.keys())}")
    
    print("\nScanning for annotation sample files...")
    sample_files = find_sample_files(samples_dir)
    print(f"Found samples for domains: {list(sample_files.keys())}")
    
    # Determine which domains to analyze
    if args.domain:
        domains = [args.domain]
    else:
        domains = sorted(set(taxonomy_files.keys()) | set(sample_files.keys()))
    
    # Analyze each domain
    for domain in domains:
        tax_files = taxonomy_files.get(domain, [])
        samp_files = sample_files.get(domain, [])
        
        if not tax_files and not samp_files:
            print(f"\nSkipping {domain}: no files found")
            continue
        
        # Aggregate types from all files for this domain
        all_ner_types = defaultdict(int)
        all_tax_types = defaultdict(int)
        
        print(f"\n--- Processing {domain} ---")
        
        if samp_files:
            print(f"Sample files: {[f.name for f in samp_files]}")
            for sf in samp_files:
                types = get_ner_types(sf)
                for t, count in types.items():
                    all_ner_types[t] += count
        
        if tax_files:
            print(f"Taxonomy files: {[f.name for f in tax_files]}")
            for tf in tax_files:
                types = get_taxonomy_types(tf)
                for t, count in types.items():
                    all_tax_types[t] += count
        
        print_comparison(domain, dict(all_ner_types), dict(all_tax_types))
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
