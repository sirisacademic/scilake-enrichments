#!/usr/bin/env python3
"""
Convert NCBI species CSV to standard TSV format.

Input CSV format:
    term,id,rank,parent_id,synonyms,variants,freq
    Homo sapiens,9606,species,9605,human,H. sapiens | Human | Humans | ...,2783958

Output TSV format:
    id, concept, synonyms, wikidata_id, wikidata_aliases, top_level, parent_id, 
    type, description, rank, variants, freq

Usage:
    python convert_species_csv_to_tsv.py \
        --input taxonomies/cancer/cancer_NCBI_species_SPECIES.csv \
        --output taxonomies/cancer/NCBI_SPECIES.tsv
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_species_csv_to_tsv(
    input_path: str,
    output_path: str,
    chunk_size: int = 100000
):
    """
    Convert NCBI species CSV to standard TSV format.
    
    Processes in chunks to handle large files.
    
    Args:
        input_path: Path to input CSV (cancer_NCBI_species_SPECIES.csv)
        output_path: Path to output TSV (NCBI_SPECIES.tsv)
        chunk_size: Number of rows per chunk
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Get total rows for progress bar
    print(f"📂 Counting rows in {input_path}...")
    total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1  # -1 for header
    print(f"   Total rows: {total_rows:,}")
    
    # Output columns in standard format
    output_columns = [
        'id', 'concept', 'synonyms', 'wikidata_id', 'wikidata_aliases',
        'top_level', 'parent_id', 'type', 'description', 'rank',
        # New columns from Linnaeus
        'variants', 'freq'
    ]
    
    # Write header
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(output_columns) + '\n')
    
    # Process in chunks
    chunks_processed = 0
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
                term = row.get('term', '')
                ncbi_id = row.get('id', '')
                rank = row.get('rank', '')
                parent_id = row.get('parent_id', '')
                synonyms = row.get('synonyms', '')
                variants = row.get('variants', '')
                freq = row.get('freq', '0')
                
                # Skip rows without term or id
                if not term or not ncbi_id:
                    continue
                
                # Build the output row
                output_rows.append({
                    'id': f'NCBI:{ncbi_id}' if ncbi_id else '',
                    'concept': term,
                    'synonyms': synonyms,
                    'wikidata_id': '',
                    'wikidata_aliases': '',
                    'top_level': 'False',
                    'parent_id': f'NCBI:{parent_id}' if parent_id else '',
                    'type': 'Species',
                    'description': '',
                    'rank': rank,
                    'variants': variants,
                    'freq': freq
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
            
            chunks_processed += 1
            pbar.update(len(chunk))
    
    # Final stats
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Conversion complete!")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Rows written: {rows_written:,}")
    print(f"   Output size: {output_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NCBI species CSV to standard TSV format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV (cancer_NCBI_species_SPECIES.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output TSV (NCBI_SPECIES.tsv)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100000,
        help="Rows per chunk (default: 100000)"
    )
    
    args = parser.parse_args()
    
    convert_species_csv_to_tsv(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
