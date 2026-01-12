#!/usr/bin/env python3
"""
Build SQLite FTS5 indices from TSV vocabulary files.

Creates disk-based full-text search indices for fast exact matching
of entity mentions against vocabularies.

Features:
- Main entities table with concept and metadata
- Separate synonym_lookup table for fast exact synonym matching
- FTS5 index for full-text search (optional)
- Frequency column for disambiguation when multiple entities match

Usage:
    # Build single index
    python build_fts5_indices.py \
        --input taxonomies/cancer/NCBI_GENE.tsv \
        --output indices/cancer/ncbi_gene.db \
        --source NCBI_Gene

    # Build single index with extra columns merged into synonyms
    python build_fts5_indices.py \
        --input taxonomies/cancer/NCBI_GENE.tsv \
        --output indices/cancer/ncbi_gene.db \
        --source NCBI_Gene \
        --extra-synonym-columns description

    # Build species index with variants and frequency
    python build_fts5_indices.py \
        --input taxonomies/cancer/NCBI_SPECIES.tsv \
        --output indices/cancer/ncbi_species.db \
        --source NCBI_Taxonomy \
        --extra-synonym-columns variants
        
    # Build diseases index with suggestions as extra synonyms
    python scripts/build_fts5_indices.py \
        -input taxonomies/cancer/DOID_DISEASE.tsv \
        -output indices/cancer/doid_disease.db \
        -source DOID \
        --extra-synonym-columns suggestions

    # Build all cancer domain indices
    python build_fts5_indices.py --build-cancer-domain
"""

import argparse
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import time
from typing import List, Optional, Set


def create_fts5_index(
    input_path: str,
    output_path: str,
    taxonomy_source: str,
    chunk_size: int = 100000,
    extra_synonym_columns: Optional[List[str]] = None
):
    """
    Build SQLite FTS5 index from TSV vocabulary file.
    
    Creates:
    - entities: Main table with id, concept, synonyms, description, type, frequency
    - synonym_lookup: Separate table with one row per synonym for fast exact matching
    - entities_fts: FTS5 virtual table for full-text search
    
    Args:
        input_path: Path to input TSV file
        output_path: Path to output SQLite database
        taxonomy_source: Source name (e.g., "NCBI_Gene", "DOID")
        chunk_size: Rows per chunk for processing
        extra_synonym_columns: Additional columns to merge into synonyms (e.g., ["description", "variants"])
    """
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    extra_synonym_columns = extra_synonym_columns or []
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing database
    if output_path.exists():
        output_path.unlink()
    
    print(f"📂 Building FTS5 index from {input_path}")
    print(f"   Output: {output_path}")
    print(f"   Source: {taxonomy_source}")
    if extra_synonym_columns:
        print(f"   Extra synonym columns: {extra_synonym_columns}")
    
    # Count rows
    print("   Counting rows...")
    total_rows = sum(1 for _ in open(input_path, 'r', encoding='utf-8')) - 1
    print(f"   Total rows: {total_rows:,}")
    
    # Create database
    conn = sqlite3.connect(str(output_path))
    cursor = conn.cursor()
    
    # Enable WAL mode for better write performance
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    
    # Create main table for metadata (with frequency column)
    cursor.execute("""
        CREATE TABLE entities (
            id TEXT PRIMARY KEY,
            concept TEXT NOT NULL,
            synonyms TEXT,
            description TEXT,
            type TEXT,
            taxonomy_source TEXT,
            frequency INTEGER DEFAULT 0
        )
    """)
    
    # Create synonym lookup table (one row per synonym)
    cursor.execute("""
        CREATE TABLE synonym_lookup (
            synonym TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            FOREIGN KEY (entity_id) REFERENCES entities(id)
        )
    """)
    
    # Create FTS5 virtual table for full-text search
    # We index concept and synonyms for searching
    cursor.execute("""
        CREATE VIRTUAL TABLE entities_fts USING fts5(
            id,
            concept,
            synonyms,
            content='entities',
            content_rowid='rowid',
            tokenize='unicode61'
        )
    """)
    
    # Create triggers to keep FTS in sync
    cursor.execute("""
        CREATE TRIGGER entities_ai AFTER INSERT ON entities BEGIN
            INSERT INTO entities_fts(rowid, id, concept, synonyms)
            VALUES (new.rowid, new.id, new.concept, new.synonyms);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER entities_ad AFTER DELETE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, id, concept, synonyms)
            VALUES ('delete', old.rowid, old.id, old.concept, old.synonyms);
        END
    """)
    
    cursor.execute("""
        CREATE TRIGGER entities_au AFTER UPDATE ON entities BEGIN
            INSERT INTO entities_fts(entities_fts, rowid, id, concept, synonyms)
            VALUES ('delete', old.rowid, old.id, old.concept, old.synonyms);
            INSERT INTO entities_fts(rowid, id, concept, synonyms)
            VALUES (new.rowid, new.id, new.concept, new.synonyms);
        END
    """)
    
    conn.commit()
    
    # Process in chunks
    entities_inserted = 0
    synonyms_inserted = 0
    start_time = time.time()
    
    print("📄 Indexing entities and synonyms...")
    
    with tqdm(total=total_rows, desc="Indexing", unit=" rows") as pbar:
        for chunk in pd.read_csv(
            input_path,
            sep='\t',
            dtype=str,
            chunksize=chunk_size,
            encoding='utf-8'
        ):
            chunk = chunk.fillna('')
            
            # Prepare rows for insertion
            entity_rows = []
            synonym_rows = []
            seen_synonym_pairs = set()  # Track (synonym, entity_id) pairs to avoid duplicates
            
            for _, row in chunk.iterrows():
                entity_id = row.get('id', '')
                concept = row.get('concept', '')
                synonyms = row.get('synonyms', '')
                description = row.get('description', '')
                entity_type = row.get('type', '')
                
                # Read frequency, default to 0 if not present or invalid
                frequency = row.get('freq', row.get('frequency', 0))
                try:
                    frequency = int(frequency) if frequency else 0
                except (ValueError, TypeError):
                    frequency = 0
                
                # Skip rows without id or concept
                if not entity_id or not concept:
                    continue
                
                # Collect all synonyms (including from extra columns)
                all_synonyms: Set[str] = set()
                
                # Parse existing synonyms (pipe-separated)
                if synonyms:
                    for syn in synonyms.split('|'):
                        syn = syn.strip()
                        if syn and syn.lower() != concept.lower():
                            all_synonyms.add(syn)
                
                # Add extra columns as synonyms (handle pipe-separated values)
                for col in extra_synonym_columns:
                    col_value = row.get(col, '').strip()
                    if col_value:
                        # Split by pipe in case column contains multiple values
                        for part in col_value.split('|'):
                            part = part.strip()
                            if part and part.lower() != concept.lower():
                                all_synonyms.add(part)
                
                # Prepare entity row (store original synonyms string for reference)
                combined_synonyms = '|'.join(sorted(all_synonyms)) if all_synonyms else ''
                entity_rows.append((
                    entity_id,
                    concept,
                    combined_synonyms,
                    description,
                    entity_type,
                    taxonomy_source,
                    frequency
                ))
                
                # Prepare synonym lookup rows (lowercase for case-insensitive matching)
                # Track seen pairs to avoid duplicates (e.g., same synonym in multiple columns)
                for syn in all_synonyms:
                    pair = (syn.lower(), entity_id)
                    if pair not in seen_synonym_pairs:
                        synonym_rows.append(pair)
                        seen_synonym_pairs.add(pair)
            
            # Batch insert entities
            if entity_rows:
                cursor.executemany(
                    """
                    INSERT OR IGNORE INTO entities 
                    (id, concept, synonyms, description, type, taxonomy_source, frequency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    entity_rows
                )
                entities_inserted += len(entity_rows)
            
            # Batch insert synonyms
            if synonym_rows:
                cursor.executemany(
                    """
                    INSERT INTO synonym_lookup (synonym, entity_id)
                    VALUES (?, ?)
                    """,
                    synonym_rows
                )
                synonyms_inserted += len(synonym_rows)
            
            conn.commit()
            pbar.update(len(chunk))
    
    # Create indices
    print("📇 Creating indices...")
    cursor.execute("CREATE INDEX idx_concept ON entities(concept COLLATE NOCASE)")
    cursor.execute("CREATE INDEX idx_type ON entities(type)")
    cursor.execute("CREATE INDEX idx_frequency ON entities(frequency DESC)")
    cursor.execute("CREATE INDEX idx_synonym ON synonym_lookup(synonym)")
    conn.commit()
    
    # Optimize FTS index
    print("🔧 Optimizing FTS index...")
    cursor.execute("INSERT INTO entities_fts(entities_fts) VALUES ('optimize')")
    conn.commit()
    
    # Switch back to normal journal mode
    cursor.execute("PRAGMA journal_mode=DELETE")
    conn.commit()
    
    # Get final stats
    cursor.execute("SELECT COUNT(*) FROM entities")
    entity_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM synonym_lookup")
    synonym_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM entities WHERE frequency > 0")
    freq_count = cursor.fetchone()[0]
    
    conn.close()
    
    elapsed = time.time() - start_time
    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
    
    print(f"\n✅ Index built successfully!")
    print(f"   Entities indexed: {entity_count:,}")
    print(f"   Synonyms indexed: {synonym_count:,}")
    print(f"   Entities with frequency: {freq_count:,}")
    print(f"   Database size: {output_size:.1f} MB")
    print(f"   Time elapsed: {elapsed:.1f} seconds")
    
    return entity_count


def build_cancer_domain_indices(base_path: str = "taxonomies/cancer", output_base: str = "indices/cancer"):
    """
    Build all FTS5 indices for the cancer domain.
    """
    
    indices_config = [
        {
            "input": "NCBI_GENE.tsv",
            "output": "ncbi_gene.db",
            "source": "NCBI_Gene",
            "extra_synonym_columns": ["description"]
        },
        {
            "input": "NCBI_SPECIES.tsv",
            "output": "ncbi_species.db",
            "source": "NCBI_Taxonomy",
            "extra_synonym_columns": ["variants"]  # Include Linnaeus variants
        },
        {
            "input": "DOID_DISEASE.tsv",
            "output": "doid_disease.db",
            "source": "DOID",
            "extra_synonym_columns": ["suggestions"]
        },
        {
            "input": "DRUGBANK_CHEMICAL.tsv",
            "output": "drugbank_chemical.db",
            "source": "DrugBank",
            "extra_synonym_columns": None
        },
        {
            "input": "BRENDA_CELLLINE.tsv",
            "output": "brenda_cellline.db",
            "source": "BRENDA",
            "extra_synonym_columns": None
        }
    ]
    
    base_path = Path(base_path)
    output_base = Path(output_base)
    
    print("=" * 60)
    print("Building FTS5 indices for cancer domain")
    print("=" * 60)
    
    results = []
    
    for config in indices_config:
        input_path = base_path / config["input"]
        output_path = output_base / config["output"]
        
        print(f"\n{'=' * 60}")
        print(f"Processing: {config['input']}")
        print("=" * 60)
        
        if not input_path.exists():
            print(f"⚠️  Skipping - file not found: {input_path}")
            results.append({
                "name": config["input"],
                "status": "skipped",
                "reason": "file not found"
            })
            continue
        
        try:
            count = create_fts5_index(
                input_path=str(input_path),
                output_path=str(output_path),
                taxonomy_source=config["source"],
                extra_synonym_columns=config.get("extra_synonym_columns")
            )
            results.append({
                "name": config["input"],
                "status": "success",
                "rows": count
            })
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "name": config["input"],
                "status": "error",
                "reason": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for result in results:
        if result["status"] == "success":
            print(f"✅ {result['name']}: {result['rows']:,} rows")
        elif result["status"] == "skipped":
            print(f"⏭️  {result['name']}: skipped ({result['reason']})")
        else:
            print(f"❌ {result['name']}: error ({result['reason']})")


def main():
    parser = argparse.ArgumentParser(
        description="Build SQLite FTS5 indices from TSV vocabulary files"
    )
    
    parser.add_argument(
        "--input", "-i",
        help="Path to input TSV file"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to output SQLite database"
    )
    parser.add_argument(
        "--source", "-s",
        help="Taxonomy source name (e.g., NCBI_Gene, DOID)"
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=100000,
        help="Rows per chunk (default: 100000)"
    )
    parser.add_argument(
        "--extra-synonym-columns",
        nargs="+",
        help="Additional columns to merge into synonyms (e.g., description variants)"
    )
    parser.add_argument(
        "--build-cancer-domain",
        action="store_true",
        help="Build all indices for cancer domain"
    )
    parser.add_argument(
        "--taxonomy-base",
        default="taxonomies/cancer",
        help="Base path for taxonomy files (default: taxonomies/cancer)"
    )
    parser.add_argument(
        "--index-base",
        default="indices/cancer",
        help="Base path for index files (default: indices/cancer)"
    )
    
    args = parser.parse_args()
    
    if args.build_cancer_domain:
        build_cancer_domain_indices(
            base_path=args.taxonomy_base,
            output_base=args.index_base
        )
    elif args.input and args.output and args.source:
        create_fts5_index(
            input_path=args.input,
            output_path=args.output,
            taxonomy_source=args.source,
            chunk_size=args.chunk_size,
            extra_synonym_columns=args.extra_synonym_columns
        )
    else:
        parser.print_help()
        print("\nError: Provide either --build-cancer-domain or (--input, --output, --source)")


if __name__ == "__main__":
    main()
