#!/usr/bin/env python3
"""
Convert DOID disease CSV to standard TSV format.

Input CSV format (from colleague):
    id, name, synonyms, ..., suggestions
    DOID:0050117, disease by infectious agent, ..., infectious diseases

Output TSV format:
    id, concept, synonyms, wikidata_id, wikidata_aliases, top_level, parent_id, 
    type, description, suggestions

The 'suggestions' column contains additional synonyms that should be included
in the FTS5 index via --extra-synonym-columns suggestions

Usage:
    python convert_disease_csv_to_tsv.py \
        --input taxonomies/cancer/cancer_disease_ontology_DISEASE_with_suggestions.csv \
        --output taxonomies/cancer/DOID_DISEASE.tsv
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_disease_csv_to_tsv(
    input_path: str,
    output_path: str,
    chunk_size: int = 50000
):
    """
    Convert DOID disease CSV to standard TSV format.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to output TSV
        chunk_size: Number of rows per chunk
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Get total rows for progress bar
    print(f"📂 Counting rows in {input_path}...")
    total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1
    print(f"   Total rows: {total_rows:,}")
    
    # First, read the CSV to inspect columns
    sample = pd.read_csv(input_path, nrows=5)
    print(f"   Input columns: {list(sample.columns)}")
    
    # Output columns in standard format
    output_columns = [
        'id', 'concept', 'synonyms', 'wikidata_id', 'wikidata_aliases',
        'top_level', 'parent_id', 'type', 'description',
        # New column for suggested synonyms
        'suggestions'
    ]
    
    # Write header
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(output_columns) + '\n')
    
    # Process in chunks
    rows_written = 0
    
    print(f"📄 Converting to TSV format...")
    
    with tqdm(total=total_rows, desc="Processing", unit=" rows") as pbar:
        for chunk in pd.read_csv(
            input_path,
            dtype=str,
            chunksize=chunk_size,
            encoding='utf-8'
        ):
            chunk = chunk.fillna('')
            
            # Build output dataframe
            output_rows = []
            
            for _, row in chunk.iterrows():
                # Map input columns to output columns
                # Adjust these mappings based on actual CSV column names
                doid_id = row.get('id', '')
                name = row.get('name', '')
                synonyms = row.get('synonyms', '')
                description = row.get('description', row.get('def', ''))
                suggestions = row.get('suggestions', '')
                
                # Skip rows without id or name
                if not doid_id or not name:
                    continue
                
                # Ensure DOID prefix
                if not doid_id.startswith('DOID:'):
                    doid_id = f'DOID:{doid_id}'
                
                output_rows.append({
                    'id': doid_id,
                    'concept': name,
                    'synonyms': synonyms,
                    'wikidata_id': '',
                    'wikidata_aliases': '',
                    'top_level': 'False',
                    'parent_id': '',
                    'type': 'Disease',
                    'description': description,
                    'suggestions': suggestions
                })
            
            # Write chunk to file
            if output_rows:
                df_out = pd.DataFrame(output_rows, columns=output_columns)
                df_out.to_csv(
                    output_path,
                    mode='a',
                    sep='\t',
                    index=False,
                    header=False,
                    encoding='utf-8'
                )
                rows_written += len(output_rows)
            
            pbar.update(len(chunk))
    
    # Final stats
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Conversion complete!")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Rows written: {rows_written:,}")
    print(f"   Output size: {output_size:.2f} MB")
    print(f"\n💡 To build the FTS5 index with suggestions as synonyms:")
    print(f"   python scripts/build_fts5_indices.py \\")
    print(f"       -i {output_path} \\")
    print(f"       -o indices/cancer/doid_disease.db \\")
    print(f"       -s DOID \\")
    print(f"       --extra-synonym-columns suggestions")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DOID disease CSV to standard TSV format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output TSV"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=50000,
        help="Rows per chunk (default: 50000)"
    )
    
    args = parser.parse_args()
    
    convert_disease_csv_to_tsv(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
