#!/usr/bin/env python3
"""
src/legal_text_reader.py

Reader for legal text JSON files (e.g., Fedlex dataset).
Converts JSON records to DataFrame format compatible with the NER pipeline.

Input format (JSONL - one JSON object per line):
{
    "rsNr": "0.101",
    "en_lawTitle": "Convention for the Protection of Human Rights...",
    "en_lawText": "Full legal text content..."
}

Output format (DataFrame):
    section_id    | section_content_expanded
    --------------|---------------------------
    0.101         | Convention for the Protection... Full legal text...

Usage:
    # As module (from pipeline.py):
    from src.legal_text_reader import parse_legal_text_file
    
    # As CLI:
    python -m src.legal_text_reader --input /path/to/file.jsonl
"""

import json
import os
from typing import List, Dict, Any, Optional, Iterator
import pandas as pd


def parse_legal_text_record(
    record: Dict[str, Any],
    include_title: bool = True
) -> List[Dict[str, str]]:
    """
    Parse a single JSON record into a section.
    
    Args:
        record: JSON record with 'rsNr', 'en_lawTitle', 'en_lawText'
        include_title: If True, prepend title to the text content
    
    Returns:
        List containing a single section dict (or empty list if invalid)
    """
    sections = []
    
    rs_nr = record.get('rsNr', '')
    if not rs_nr:
        return sections
    
    title = record.get('en_lawTitle', '').strip()
    text = record.get('en_lawText', '').strip()
    
    if not text and not title:
        return sections
    
    # Combine title and text, normalizing whitespace
    if include_title and title and text:
        # Add period after title if it doesn't end with punctuation
        if title[-1] not in '.!?:':
            combined_text = f"{title}. {text}"
        else:
            combined_text = f"{title} {text}"
    elif title:
        combined_text = title
    else:
        combined_text = text
    
    # Normalize whitespace (replace newlines and multiple spaces with single space)
    combined_text = ' '.join(combined_text.split())
    
    sections.append({
        'section_id': rs_nr,
        'section_content_expanded': combined_text,
        'rsNr': rs_nr,
        'section_type': 'legal_text',
        'title': title
    })
    
    return sections


def iter_json_records(filepath: str) -> Iterator[Dict[str, Any]]:
    """
    Iterate over JSON records in a file.
    Supports both JSONL (one record per line) and JSON array format.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        
        if first_char == '[':
            # JSON array format
            data = json.load(f)
            for record in data:
                yield record
        else:
            # JSONL format (one JSON object per line)
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"⚠️  JSON error at line {line_num}: {e}")
                    continue


def parse_legal_text_file(
    filepath: str,
    include_title: bool = True,
    logger=None
) -> pd.DataFrame:
    """
    Parse a legal text JSON file into a DataFrame.
    
    Args:
        filepath: Path to the JSON/JSONL file
        include_title: If True, prepend title to the text content
        logger: Optional logger
        
    Returns:
        DataFrame with columns: section_id, section_content_expanded, rsNr, section_type, title
    """
    if logger:
        logger.info(f"📂 Parsing legal text file: {os.path.basename(filepath)}")
    
    all_sections = []
    record_count = 0
    
    for record in iter_json_records(filepath):
        record_count += 1
        sections = parse_legal_text_record(record, include_title=include_title)
        all_sections.extend(sections)
    
    df = pd.DataFrame(all_sections)
    
    if logger:
        logger.info(f"   Records: {record_count}, Sections created: {len(df)}")
        if not df.empty:
            avg_len = df['section_content_expanded'].str.len().mean()
            max_len = df['section_content_expanded'].str.len().max()
            logger.info(f"   Avg text length: {avg_len:.0f} chars, Max: {max_len} chars")
    
    return df


def parse_legal_text_directory(
    input_dir: str,
    pattern: Optional[str] = None,
    include_title: bool = True,
    logger=None
) -> pd.DataFrame:
    """
    Parse all legal text JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        pattern: Optional filename pattern filter (e.g., 'fedlex' will match 'fedlex*.json')
        include_title: If True, prepend title to the text content
        logger: Optional logger
        
    Returns:
        Combined DataFrame from all matching files
    """
    all_dfs = []
    
    for filename in os.listdir(input_dir):
        if not (filename.endswith('.json') or filename.endswith('.jsonl')):
            continue
        
        # Filter by pattern if specified
        if pattern and pattern.lower() not in filename.lower():
            continue
        
        filepath = os.path.join(input_dir, filename)
        df = parse_legal_text_file(filepath, include_title=include_title, logger=logger)
        if not df.empty:
            df['source_file'] = filename
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame(columns=['section_id', 'section_content_expanded', 'rsNr', 'section_type', 'title'])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if logger:
        logger.info(f"✅ Total sections loaded: {len(combined)}")
    
    return combined


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse legal text JSON files")
    parser.add_argument("--input", required=True, help="Path to JSON file or directory")
    parser.add_argument("--pattern", help="Filename pattern filter (e.g., fedlex)")
    parser.add_argument("--no-title", action="store_true", help="Don't include title in text")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--preview", type=int, default=5, help="Number of rows to preview")
    parser.add_argument("--stats", action="store_true", help="Show detailed statistics")
    args = parser.parse_args()
    
    include_title = not args.no_title
    
    if os.path.isfile(args.input):
        df = parse_legal_text_file(args.input, include_title=include_title)
    else:
        df = parse_legal_text_directory(args.input, pattern=args.pattern, include_title=include_title)
    
    print(f"\n📊 Parsed {len(df)} sections")
    print(f"   Include title: {include_title}")
    
    if args.stats and not df.empty:
        lengths = df['section_content_expanded'].str.len()
        print(f"\n📈 Text length statistics:")
        print(f"   Min: {lengths.min():,} chars")
        print(f"   Max: {lengths.max():,} chars")
        print(f"   Mean: {lengths.mean():,.0f} chars")
        print(f"   Median: {lengths.median():,.0f} chars")
        print(f"   Over 100K: {(lengths > 100000).sum()}")
        print(f"   Over 1M: {(lengths > 1000000).sum()}")
    
    print(f"\nColumn types:")
    print(df.dtypes)
    
    print(f"\nPreview ({args.preview} rows):")
    pd.set_option('display.max_colwidth', 80)
    preview_cols = ['section_id', 'title', 'section_content_expanded']
    preview_cols = [c for c in preview_cols if c in df.columns]
    print(df[preview_cols].head(args.preview))
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✅ Saved to {args.output}")
