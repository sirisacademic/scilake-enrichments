#!/usr/bin/env python3
"""
Test FlashText directly on the problematic section text.
"""

import pandas as pd
from flashtext import KeywordProcessor

# Load the section
csv_path = "outputs/energy-all-ft/sections/doi_________::81e19bdb07b2b3070af2bef2ef71e0a0::661c354a59c8c14f322dec9c8f2fd1d1_sections.csv"

df = pd.read_csv(csv_path, dtype=str)
section_id = "http://scilake-project.eu/res/4a47de2b#offset_34931_41754"
row = df[df['section_id'] == section_id].iloc[0]

expanded_text = row['section_content_expanded']
original_text = row.get('section_content', '')

print(f"Expanded text length: {len(expanded_text)}")
print(f"Original text length: {len(original_text)}")

# Create a simple keyword processor
kp = KeywordProcessor(case_sensitive=False)
kp.add_keyword("energy", {"concept": "Energy"})
kp.add_keyword("solar energy", {"concept": "Solar energy"})
kp.add_keyword("renewable energy", {"concept": "Renewable energy"})

print("\n" + "="*80)
print("Testing on EXPANDED text:")
print("="*80)

matches = kp.extract_keywords(expanded_text, span_info=True)
print(f"\nFound {len(matches)} matches")

for metadata, start, end in matches[:20]:
    actual_text = expanded_text[start:end]
    print(f"  [{start:4d}-{end:4d}] '{actual_text}' -> {metadata['concept']}")
    
    # Check if actual text matches what we expect
    if actual_text.lower() != metadata['concept'].lower():
        # Check surrounding context
        ctx_start = max(0, start - 10)
        ctx_end = min(len(expanded_text), end + 10)
        context = expanded_text[ctx_start:ctx_end]
        print(f"           ⚠️  MISMATCH! Context: ...{context}...")

print("\n" + "="*80)
print("Testing on ORIGINAL text:")
print("="*80)

if original_text:
    matches_orig = kp.extract_keywords(original_text, span_info=True)
    print(f"\nFound {len(matches_orig)} matches")
    
    for metadata, start, end in matches_orig[:20]:
        actual_text = original_text[start:end]
        print(f"  [{start:4d}-{end:4d}] '{actual_text}' -> {metadata['concept']}")

# Now let's check specific positions manually
print("\n" + "="*80)
print("Manual check of specific offsets:")
print("="*80)

# Look for "Solar Energy" manually
search_term = "Solar Energy"
idx = expanded_text.find(search_term)
if idx != -1:
    print(f"\n'{search_term}' found at offset {idx} in expanded text")
    print(f"  Text at {idx}:{idx+len(search_term)}: '{expanded_text[idx:idx+len(search_term)]}'")

idx = expanded_text.lower().find(search_term.lower())
if idx != -1:
    print(f"\n'{search_term}' (case-insensitive) found at offset {idx}")
    print(f"  Text at {idx}:{idx+len(search_term)}: '{expanded_text[idx:idx+len(search_term)]}'")

# Check what's around offset 1088
print(f"\n\nText around offset 1088-1100:")
print(f"  expanded[1078:1110]: '{expanded_text[1078:1110]}'")
print(f"  expanded[1088:1100]: '{expanded_text[1088:1100]}'")

# Check encoding
print(f"\n\nChecking for encoding issues:")
print(f"  Chars at 1085-1095: {[expanded_text[i] for i in range(1085, 1095)]}")
print(f"  Ord values: {[ord(expanded_text[i]) for i in range(1085, 1095)]}")
