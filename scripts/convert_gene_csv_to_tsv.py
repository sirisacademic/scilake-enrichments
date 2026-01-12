#!/usr/bin/env python3
"""
Convert NCBI gene CSV to standard TSV format.

Input:  cancer_NCBI_gene.csv (64M rows)
Output: NCBI_GENE.tsv (same format as other taxonomy files)

Usage:
    python convert_gene_csv_to_tsv.py \
        --input taxonomies/cancer/cancer_NCBI_gene.csv \
        --output taxonomies/cancer/NCBI_GENE.tsv
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def convert_gene_csv_to_tsv(
    input_path: str,
    output_path: str,
    chunk_size: int = 500000
):
    """
    Convert NCBI gene CSV to standard TSV format.
    
    Processes in chunks to handle large files.
    
    Args:
        input_path: Path to input CSV (cancer_NCBI_gene.csv)
        output_path: Path to output TSV (NCBI_GENE.tsv)
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
        'top_level', 'parent_id', 'type', 'description',
        # Extra columns preserved from original
        'type_of_gene', 'organism', 'organism_id'
    ]
    
    # Write header
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\t'.join(output_columns) + '\n')
    
    # Process in chunks
    chunks_processed = 0
    rows_written = 0
    
    print(f"🔄 Converting to TSV format...")
    
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
                gene_id = row.get('GeneID', '')
                symbol = row.get('Symbol', '')
                synonyms_orig = row.get('Synonyms', '')
                description = row.get('Description', '')
                type_of_gene = row.get('Type_of_gene', '')
                organism = row.get('Organism', '')
                organism_id = row.get('Organism_ID', '')
                
                # Skip rows without symbol
                if not symbol:
                    continue
                
                # Build synonyms: original synonyms + description (if different from symbol)
                synonyms_parts = []
                if synonyms_orig:
                    synonyms_parts.append(synonyms_orig)
                if description and description.lower() != symbol.lower():
                    synonyms_parts.append(description)
                
                synonyms_combined = '|'.join(synonyms_parts) if synonyms_parts else ''
                
                output_rows.append({
                    'id': f'NCBI:{gene_id}' if gene_id else '',
                    'concept': symbol,
                    'synonyms': synonyms_combined,
                    'wikidata_id': '',
                    'wikidata_aliases': '',
                    'top_level': 'False',
                    'parent_id': '',
                    'type': 'Gene',
                    'description': description,
                    'type_of_gene': type_of_gene,
                    'organism': organism,
                    'organism_id': organism_id
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
    output_size = output_path.stat().st_size / (1024 * 1024 * 1024)  # GB
    
    print(f"\n✅ Conversion complete!")
    print(f"   Input:  {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Rows written: {rows_written:,}")
    print(f"   Output size: {output_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NCBI gene CSV to standard TSV format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input CSV (cancer_NCBI_gene.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to output TSV (NCBI_GENE.tsv)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=500000,
        help="Rows per chunk (default: 500000)"
    )
    
    args = parser.parse_args()
    
    convert_gene_csv_to_tsv(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
