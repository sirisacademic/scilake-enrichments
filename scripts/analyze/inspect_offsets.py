#!/usr/bin/env python3
"""
Inspect entity offsets to diagnose mismatch between mention text and actual text at offset.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Problematic samples to investigate
SAMPLES = [
    {"id": "4a47de2b#offset_34931_41754#4877-4883", "mention": "gy sys", "start": 4877, "end": 4883},
    {"id": "4a47de2b#offset_34931_41754#3118-3130", "mention": "r Energy Pot", "start": 3118, "end": 3130},
    {"id": "4a47de2b#offset_34931_41754#1303-1319", "mention": "wable Energy Lab", "start": 1303, "end": 1319},
]

def extract_section_id(sample_id: str) -> str:
    """Extract section_id from sample id format: hash#offset_X_Y#start-end"""
    parts = sample_id.rsplit('#', 1)  # Split from right to get section part
    return f"http://scilake-project.eu/res/{parts[0]}"

def find_file_with_section(sections_dir: Path, section_id: str) -> Path:
    """Find the CSV file containing a specific section_id."""
    for csv_file in sections_dir.glob("*_sections.csv"):
        try:
            df = pd.read_csv(csv_file, dtype=str)
            if section_id in df['section_id'].values:
                return csv_file
        except Exception:
            continue
    return None

def find_el_file_with_section(el_dir: Path, section_id: str) -> Path:
    """Find the JSONL file containing a specific section_id."""
    for jsonl_file in el_dir.glob("*.jsonl"):
        try:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('section_id') == section_id:
                        return jsonl_file
        except Exception:
            continue
    return None

def main():
    if len(sys.argv) < 3:
        print("Usage: python inspect_offsets.py <sections_dir> <el_dir>")
        print("Example: python inspect_offsets.py outputs/energy-all-ft/sections outputs/energy-all-ft/el")
        sys.exit(1)
    
    sections_dir = Path(sys.argv[1])
    el_dir = Path(sys.argv[2])
    
    print("=" * 80)
    print("OFFSET INSPECTION REPORT")
    print("=" * 80)
    
    for sample in SAMPLES:
        sample_id = sample["id"]
        expected_mention = sample["mention"]
        start = sample["start"]
        end = sample["end"]
        
        section_id = extract_section_id(sample_id)
        
        print(f"\n{'='*80}")
        print(f"Sample ID: {sample_id}")
        print(f"Section ID: {section_id}")
        print(f"Expected mention: '{expected_mention}'")
        print(f"Offsets: {start}-{end}")
        print("-" * 80)
        
        # Find and load sections file
        csv_file = find_file_with_section(sections_dir, section_id)
        if csv_file:
            print(f"Found in sections file: {csv_file.name}")
            df = pd.read_csv(csv_file, dtype=str)
            row = df[df['section_id'] == section_id].iloc[0]
            
            expanded_text = row.get('section_content_expanded', '')
            original_text = row.get('section_content', '')
            
            print(f"\nExpanded text length: {len(expanded_text)}")
            print(f"Original text length: {len(original_text) if original_text else 'N/A'}")
            
            # Check what's at the offset in expanded text
            if expanded_text and start < len(expanded_text):
                actual_at_offset = expanded_text[start:end]
                print(f"\n[EXPANDED] Text at offset {start}:{end}: '{actual_at_offset}'")
                
                # Show context around the offset
                ctx_start = max(0, start - 30)
                ctx_end = min(len(expanded_text), end + 30)
                context = expanded_text[ctx_start:ctx_end]
                print(f"[EXPANDED] Context: ...{context}...")
            else:
                print(f"\n[EXPANDED] Offset {start}:{end} is OUT OF BOUNDS (text length: {len(expanded_text)})")
            
            # Check what's at the offset in original text (if available)
            if original_text and start < len(original_text):
                actual_at_offset_orig = original_text[start:end]
                print(f"\n[ORIGINAL] Text at offset {start}:{end}: '{actual_at_offset_orig}'")
                
                ctx_start = max(0, start - 30)
                ctx_end = min(len(original_text), end + 30)
                context_orig = original_text[ctx_start:ctx_end]
                print(f"[ORIGINAL] Context: ...{context_orig}...")
            
            # Try to find the expected mention in the expanded text
            if expanded_text:
                # Search for full expected concept names
                search_terms = ["energy sys", "Energy Pot", "Renewable Energy Lab", "Solar Energy", expected_mention]
                print(f"\n[SEARCH] Looking for related terms in expanded text:")
                for term in search_terms:
                    idx = expanded_text.find(term)
                    if idx != -1:
                        print(f"  Found '{term}' at offset {idx}")
        else:
            print(f"Section file NOT FOUND")
        
        # Find and check EL file
        print("\n" + "-" * 40)
        el_file = find_el_file_with_section(el_dir, section_id)
        if el_file:
            print(f"Found in EL file: {el_file.name}")
            with open(el_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    if data.get('section_id') == section_id:
                        # Find matching entity
                        for ent in data.get('entities', []):
                            if ent.get('start') == start and ent.get('end') == end:
                                print(f"\n[EL ENTITY] Found entity:")
                                print(f"  text: '{ent.get('text')}'")
                                print(f"  start: {ent.get('start')}")
                                print(f"  end: {ent.get('end')}")
                                print(f"  model: {ent.get('model')}")
                                print(f"  linking: {ent.get('linking')}")
                        break
        else:
            print(f"EL file NOT FOUND")

if __name__ == "__main__":
    main()
