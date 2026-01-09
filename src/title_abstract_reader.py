#!/usr/bin/env python3
"""
src/title_abstract_reader.py

Reader for title/abstract JSON files.
Converts JSON records to DataFrame format compatible with the NER pipeline.

Input format (JSONL - one JSON object per line):
{
    "abstracts": ["abstract1", "abstract2", ...],
    "oaireid": "50|doi_dedup___::...",
    "pids": [{"scheme": "doi", "value": "..."}],
    "resulttype": "publication",
    "titles": ["title1", "title2", ...]
}

Output format (DataFrame) - Combined mode (default):
    section_id                              | section_content_expanded
    ----------------------------------------|---------------------------
    50|doi_dedup___::abc123                  | First title. First abstract text

Output format (DataFrame) - Separate mode:
    section_id                              | section_content_expanded
    ----------------------------------------|---------------------------
    50|doi_dedup___::abc123#title           | First title text
    50|doi_dedup___::abc123#abstract        | First abstract text

Usage:
    # As module (from pipeline.py):
    from src.title_abstract_reader import parse_title_abstract_file
    
    # As CLI:
    python -m src.title_abstract_reader --input /path/to/file.json
    python -m src.title_abstract_reader --input /path/to/file.json --separate  # Separate title/abstract
"""

import json
import os
from typing import List, Dict, Any, Optional, Iterator
import pandas as pd


def parse_title_abstract_record(
    record: Dict[str, Any],
    combine_sections: bool = True
) -> List[Dict[str, str]]:
    """
    Parse a single JSON record into sections.
    
    Takes the first element from titles and abstracts lists.
    
    Args:
        record: JSON record with 'oaireid', 'titles', 'abstracts'
        combine_sections: If True, combine title and abstract into single section.
                         If False, create separate sections for title and abstract.
    
    Returns:
        List of section dicts with section_id and section_content_expanded.
    """
    sections = []
    oaireid = record.get('oaireid', '')
    
    if not oaireid:
        return sections
    
    # Extract title (take first element)
    title_text = ""
    titles = record.get('titles', [])
    if titles and isinstance(titles, list) and len(titles) > 0:
        title = titles[0]
        if title and isinstance(title, str) and title.strip():
            title_text = title.strip()
    
    # Extract abstract (take first element)
    abstract_text = ""
    abstracts = record.get('abstracts', [])
    if abstracts and isinstance(abstracts, list) and len(abstracts) > 0:
        abstract = abstracts[0]
        if abstract and isinstance(abstract, str) and abstract.strip():
            abstract_text = abstract.strip()
    
    if combine_sections:
        # Combined mode: single section with "title. abstract"
        if title_text or abstract_text:
            if title_text and abstract_text:
                # Combine title and abstract
                # Add period after title if it doesn't end with punctuation
                if title_text[-1] not in '.!?':
                    combined_text = f"{title_text}. {abstract_text}"
                else:
                    combined_text = f"{title_text} {abstract_text}"
            elif title_text:
                combined_text = title_text
            else:
                combined_text = abstract_text
            
            sections.append({
                'section_id': oaireid,
                'section_content_expanded': combined_text,
                'oaireid': oaireid,
                'section_type': 'title_abstract'
            })
    else:
        # Separate mode: one section per title/abstract
        if title_text:
            sections.append({
                'section_id': f"{oaireid}#title",
                'section_content_expanded': title_text,
                'oaireid': oaireid,
                'section_type': 'title'
            })
        
        if abstract_text:
            sections.append({
                'section_id': f"{oaireid}#abstract",
                'section_content_expanded': abstract_text,
                'oaireid': oaireid,
                'section_type': 'abstract'
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


def parse_title_abstract_file(
    filepath: str,
    combine_sections: bool = True,
    logger=None
) -> pd.DataFrame:
    """
    Parse a title/abstract JSON file into a DataFrame.
    
    Args:
        filepath: Path to the JSON/JSONL file
        combine_sections: If True (default), combine title and abstract into single section.
        logger: Optional logger
        
    Returns:
        DataFrame with columns: section_id, section_content_expanded, oaireid, section_type
    """
    if logger:
        mode = "combined" if combine_sections else "separate"
        logger.info(f"📂 Parsing title/abstract file ({mode} mode): {os.path.basename(filepath)}")
    
    all_sections = []
    record_count = 0
    titles_found = 0
    abstracts_found = 0
    
    for record in iter_json_records(filepath):
        record_count += 1
        
        # Count titles and abstracts for stats
        titles = record.get('titles', [])
        abstracts = record.get('abstracts', [])
        if titles and len(titles) > 0 and titles[0]:
            titles_found += 1
        if abstracts and len(abstracts) > 0 and abstracts[0]:
            abstracts_found += 1
        
        sections = parse_title_abstract_record(record, combine_sections=combine_sections)
        all_sections.extend(sections)
    
    df = pd.DataFrame(all_sections)
    
    if logger:
        logger.info(f"   Records: {record_count}, Titles: {titles_found}, Abstracts: {abstracts_found}")
        logger.info(f"   Sections created: {len(df)}")
    
    return df


def parse_title_abstract_directory(
    input_dir: str,
    domain: Optional[str] = None,
    combine_sections: bool = True,
    logger=None
) -> pd.DataFrame:
    """
    Parse all title/abstract JSON files in a directory.
    
    Args:
        input_dir: Directory containing JSON files
        domain: Optional domain filter (e.g., 'ccam' will match 'ccam_titleabstract.json')
        combine_sections: If True (default), combine title and abstract into single section.
        logger: Optional logger
        
    Returns:
        Combined DataFrame from all matching files
    """
    all_dfs = []
    
    for filename in os.listdir(input_dir):
        if not filename.endswith('.json'):
            continue
        
        # Filter by domain if specified
        if domain and not filename.lower().startswith(domain.lower()):
            continue
        
        filepath = os.path.join(input_dir, filename)
        df = parse_title_abstract_file(filepath, combine_sections=combine_sections, logger=logger)
        if not df.empty:
            df['source_file'] = filename
            all_dfs.append(df)
    
    if not all_dfs:
        return pd.DataFrame(columns=['section_id', 'section_content_expanded', 'oaireid', 'section_type'])
    
    combined = pd.concat(all_dfs, ignore_index=True)
    
    if logger:
        logger.info(f"✅ Total sections loaded: {len(combined)}")
    
    return combined


# ============================================================
# CLI for testing
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse title/abstract JSON files")
    parser.add_argument("--input", required=True, help="Path to JSON file or directory")
    parser.add_argument("--domain", help="Domain filter (e.g., ccam, energy)")
    parser.add_argument("--separate", action="store_true", 
                       help="Create separate sections for title and abstract (default: combined)")
    parser.add_argument("--output", help="Output CSV path (optional)")
    parser.add_argument("--preview", type=int, default=5, help="Number of rows to preview")
    args = parser.parse_args()
    
    combine = not args.separate
    
    if os.path.isfile(args.input):
        df = parse_title_abstract_file(args.input, combine_sections=combine)
    else:
        df = parse_title_abstract_directory(args.input, domain=args.domain, combine_sections=combine)
    
    print(f"\n📊 Parsed {len(df)} sections")
    print(f"   Mode: {'combined' if combine else 'separate'}")
    print(f"\nColumn types:")
    print(df.dtypes)
    
    print(f"\nSection type distribution:")
    if 'section_type' in df.columns:
        print(df['section_type'].value_counts())
    
    print(f"\nPreview ({args.preview} rows):")
    pd.set_option('display.max_colwidth', 80)
    print(df.head(args.preview))
    
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\n✅ Saved to {args.output}")
