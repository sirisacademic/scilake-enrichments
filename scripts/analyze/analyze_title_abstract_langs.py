#!/usr/bin/env python3
"""
Quick analysis script to check language patterns in title/abstract JSON files.
Checks if the last element in abstracts/titles lists is consistently English.
"""

import json
import os
import sys
from collections import Counter

# Try to use langdetect if available, otherwise use simple heuristics
try:
    from langdetect import detect, LangDetectException
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    print("⚠️  langdetect not installed, using simple heuristics")


def simple_language_detect(text):
    """Simple heuristic-based language detection."""
    if not text or len(text) < 20:
        return "unknown"
    
    # Check for Arabic script
    if any('\u0600' <= c <= '\u06FF' for c in text):
        return "ar"
    
    # Check for Chinese characters
    if any('\u4e00' <= c <= '\u9fff' for c in text):
        return "zh"
    
    # Check for common Spanish words
    spanish_markers = ['el ', 'la ', 'los ', 'las ', 'de ', 'del ', 'en ', 'que ', 'es ', 'un ', 'una ', 'por ', 'para ']
    spanish_count = sum(1 for m in spanish_markers if m in text.lower())
    
    # Check for common French words
    french_markers = ['le ', 'la ', 'les ', 'de ', 'du ', 'des ', 'un ', 'une ', 'est ', 'sont ', 'dans ', 'pour ', "l'", "d'"]
    french_count = sum(1 for m in french_markers if m in text.lower())
    
    # Check for common English words
    english_markers = ['the ', 'is ', 'are ', 'was ', 'were ', 'have ', 'has ', 'been ', 'will ', 'would ', 'could ', 'should ', 'this ', 'that ', 'with ', 'from ']
    english_count = sum(1 for m in english_markers if m in text.lower())
    
    # Check for common German words
    german_markers = ['der ', 'die ', 'das ', 'und ', 'ist ', 'sind ', 'wird ', 'werden ', 'für ', 'mit ', 'auf ']
    german_count = sum(1 for m in german_markers if m in text.lower())
    
    scores = {
        'en': english_count,
        'es': spanish_count,
        'fr': french_count,
        'de': german_count
    }
    
    max_lang = max(scores, key=scores.get)
    if scores[max_lang] >= 3:
        return max_lang
    return "unknown"


def detect_language(text):
    """Detect language of text."""
    if not text or len(text.strip()) < 10:
        return "unknown"
    
    if HAS_LANGDETECT:
        try:
            return detect(text)
        except LangDetectException:
            return "unknown"
    else:
        return simple_language_detect(text)


def analyze_json_file(filepath):
    """Analyze a single JSON file with multiple records."""
    results = {
        'total_records': 0,
        'records_with_abstracts': 0,
        'records_with_titles': 0,
        'last_abstract_langs': Counter(),
        'first_abstract_langs': Counter(),
        'last_title_langs': Counter(),
        'first_title_langs': Counter(),
        'abstract_list_lengths': Counter(),
        'title_list_lengths': Counter(),
        'samples': []
    }
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  ⚠️  JSON error at line {line_num}: {e}")
                continue
            
            results['total_records'] += 1
            oaireid = record.get('oaireid', 'unknown')
            
            # Analyze abstracts
            abstracts = record.get('abstracts', [])
            if abstracts:
                results['records_with_abstracts'] += 1
                results['abstract_list_lengths'][len(abstracts)] += 1
                
                # First abstract language
                first_lang = detect_language(abstracts[0])
                results['first_abstract_langs'][first_lang] += 1
                
                # Last abstract language
                last_lang = detect_language(abstracts[-1])
                results['last_abstract_langs'][last_lang] += 1
                
                # Store sample if interesting (multiple abstracts)
                if len(abstracts) > 1 and len(results['samples']) < 3:
                    results['samples'].append({
                        'oaireid': oaireid,
                        'num_abstracts': len(abstracts),
                        'first_lang': first_lang,
                        'last_lang': last_lang,
                        'first_preview': abstracts[0][:100] + '...' if len(abstracts[0]) > 100 else abstracts[0],
                        'last_preview': abstracts[-1][:100] + '...' if len(abstracts[-1]) > 100 else abstracts[-1]
                    })
            
            # Analyze titles
            titles = record.get('titles', [])
            if titles:
                results['records_with_titles'] += 1
                results['title_list_lengths'][len(titles)] += 1
                
                first_lang = detect_language(titles[0])
                results['first_title_langs'][first_lang] += 1
                
                last_lang = detect_language(titles[-1])
                results['last_title_langs'][last_lang] += 1
    
    return results


def print_results(filepath, results):
    """Print analysis results."""
    print(f"\n{'='*60}")
    print(f"📊 Analysis: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    print(f"\n📈 Overview:")
    print(f"   Total records: {results['total_records']}")
    print(f"   Records with abstracts: {results['records_with_abstracts']}")
    print(f"   Records with titles: {results['records_with_titles']}")
    
    print(f"\n📝 Abstract list lengths:")
    for length, count in sorted(results['abstract_list_lengths'].items()):
        pct = 100 * count / results['records_with_abstracts'] if results['records_with_abstracts'] else 0
        print(f"   {length} abstract(s): {count} ({pct:.1f}%)")
    
    print(f"\n🌐 FIRST abstract language:")
    for lang, count in results['first_abstract_langs'].most_common(10):
        pct = 100 * count / results['records_with_abstracts'] if results['records_with_abstracts'] else 0
        print(f"   {lang}: {count} ({pct:.1f}%)")
    
    print(f"\n🌐 LAST abstract language:")
    for lang, count in results['last_abstract_langs'].most_common(10):
        pct = 100 * count / results['records_with_abstracts'] if results['records_with_abstracts'] else 0
        marker = " ✅" if lang == 'en' else ""
        print(f"   {lang}: {count} ({pct:.1f}%){marker}")
    
    print(f"\n📑 Title list lengths:")
    for length, count in sorted(results['title_list_lengths'].items()):
        pct = 100 * count / results['records_with_titles'] if results['records_with_titles'] else 0
        print(f"   {length} title(s): {count} ({pct:.1f}%)")
    
    print(f"\n🌐 FIRST title language:")
    for lang, count in results['first_title_langs'].most_common(10):
        pct = 100 * count / results['records_with_titles'] if results['records_with_titles'] else 0
        print(f"   {lang}: {count} ({pct:.1f}%)")
    
    print(f"\n🌐 LAST title language:")
    for lang, count in results['last_title_langs'].most_common(10):
        pct = 100 * count / results['records_with_titles'] if results['records_with_titles'] else 0
        marker = " ✅" if lang == 'en' else ""
        print(f"   {lang}: {count} ({pct:.1f}%){marker}")
    
    if results['samples']:
        print(f"\n📋 Sample records with multiple abstracts:")
        for i, sample in enumerate(results['samples'], 1):
            print(f"\n   Sample {i}:")
            print(f"   ID: {sample['oaireid']}")
            print(f"   Num abstracts: {sample['num_abstracts']}")
            print(f"   First lang: {sample['first_lang']}")
            print(f"   Last lang: {sample['last_lang']}")
            print(f"   First preview: {sample['first_preview']}")
            print(f"   Last preview: {sample['last_preview']}")


def main():
    # Default path - can be overridden via command line
    default_dir = "/root/scilake-enrichments/data/title_abstract_json"
    
    if len(sys.argv) > 1:
        paths = sys.argv[1:]
    elif os.path.exists(default_dir):
        paths = [os.path.join(default_dir, f) for f in os.listdir(default_dir) if f.endswith('.json')]
    else:
        print(f"Usage: python {sys.argv[0]} <json_file_or_directory>")
        print(f"Default directory not found: {default_dir}")
        sys.exit(1)
    
    if not paths:
        print("No JSON files found")
        sys.exit(1)
    
    print(f"🔍 Analyzing {len(paths)} file(s)...")
    
    for filepath in sorted(paths):
        if os.path.isfile(filepath):
            results = analyze_json_file(filepath)
            print_results(filepath, results)
    
    print(f"\n{'='*60}")
    print("✅ Analysis complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
