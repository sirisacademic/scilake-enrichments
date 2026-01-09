#!/usr/bin/env python3
"""
Analyze NER output to identify diseases needing synonym enrichment.

This script:
1. Extracts all disease mentions from NER output
2. Tests each unique term against FTS5 index
3. Identifies unlinked terms that could be mapped to existing DOID entries
4. Generates recommendations for synonym additions

Usage:
    python analyze_unlinked_diseases.py \
        --ner-dir outputs/cancer-full/ner \
        --index indices/cancer/doid_disease.db \
        --top 500
"""

import argparse
import json
import sqlite3
import re
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Known generic terms that are NOT diseases (should be skipped)
# =============================================================================
SKIP_TERMS = {
    # Processes / mechanisms
    "inflammatory", "inflammation", "toxicity", "cytotoxicity", 
    "ferroptosis", "apoptosis", "autophagy", "necrosis",
    "hypoxia", "ischemia", "oxidative stress",
    
    # Symptoms / outcomes
    "death", "pain", "fatigue", "nausea", "vomiting",
    "shortness of breath", "dyspnea", "cough", "fever",
    "weakness", "weight loss", "loss of appetite",
    
    # Generic clinical terms
    "infection", "serious illness", "illness", "disease",
    "syndrome", "disorder", "condition", "complication",
    
    # Pathological features (not diseases)
    "lvi", "lymphovascular invasion", "metastasis", "metastases",
    "tumor", "tumors", "tumour", "tumours", "neoplasm",
    "lesion", "lesions", "mass", "nodule",
    
    # Lab findings
    "hypotension", "hypertension", "hypocalcemia", "hypercalcemia",
    "anemia", "neutropenia", "thrombocytopenia",
}


def load_disease_mentions(ner_dir: str) -> Counter:
    """Load all disease mentions from NER output files."""
    
    ner_path = Path(ner_dir)
    diseases = Counter()
    
    files = list(ner_path.glob("*.jsonl"))
    print(f"📂 Loading disease mentions from {len(files)} files...")
    
    for f in files:
        try:
            for line in open(f, encoding='utf-8'):
                data = json.loads(line)
                for ent in data.get('entities', []):
                    if ent.get('entity', '').lower() == 'disease':
                        text = ent.get('text', '').strip().lower()
                        if text:
                            diseases[text] += 1
        except Exception as e:
            print(f"   ⚠️ Error reading {f.name}: {e}")
    
    print(f"   Found {sum(diseases.values()):,} total mentions")
    print(f"   Found {len(diseases):,} unique terms")
    
    return diseases


def check_fts5_match(conn: sqlite3.Connection, term: str) -> Optional[Tuple[str, str]]:
    """Check if term matches in FTS5 index. Returns (id, concept) or None."""
    
    # Check exact concept match
    cursor = conn.execute(
        """
        SELECT id, concept FROM entities
        WHERE concept = ? COLLATE NOCASE
        LIMIT 1
        """,
        (term,)
    )
    result = cursor.fetchone()
    if result:
        return result
    
    # Check synonym match
    cursor = conn.execute(
        """
        SELECT id, concept, synonyms FROM entities
        WHERE synonyms = ? COLLATE NOCASE
           OR synonyms LIKE ? COLLATE NOCASE
           OR synonyms LIKE ? COLLATE NOCASE
           OR synonyms LIKE ? COLLATE NOCASE
        LIMIT 1
        """,
        (term, f"{term}|%", f"%|{term}|%", f"%|{term}")
    )
    result = cursor.fetchone()
    if result:
        return (result[0], result[1])
    
    return None


def search_candidates(conn: sqlite3.Connection, term: str, limit: int = 5) -> List[Tuple[str, str]]:
    """Search for potential matches using partial matching."""
    
    candidates = []
    
    # Extract key words from term
    words = re.findall(r'\b\w+\b', term.lower())
    
    # Skip very short or generic words
    keywords = [w for w in words if len(w) > 3 and w not in {
        'cell', 'type', 'with', 'from', 'like', 'stage', 'grade',
        'early', 'late', 'high', 'low', 'primary', 'secondary',
        'acute', 'chronic', 'advanced', 'metastatic', 'recurrent'
    }]
    
    if not keywords:
        return []
    
    # Search for each keyword
    for keyword in keywords[:2]:  # Limit to first 2 keywords
        cursor = conn.execute(
            """
            SELECT id, concept FROM entities
            WHERE concept LIKE ? COLLATE NOCASE
            LIMIT ?
            """,
            (f"%{keyword}%", limit)
        )
        for row in cursor.fetchall():
            if row not in candidates:
                candidates.append(row)
    
    return candidates[:limit]


def classify_unlinked_term(term: str, candidates: List[Tuple[str, str]]) -> str:
    """Classify why a term is unlinked."""
    
    term_lower = term.lower()
    
    # Check if it's in skip list
    if term_lower in SKIP_TERMS:
        return "SKIP: Generic/process term"
    
    # Check if it looks like an abbreviation
    if len(term) <= 5 and term.isupper():
        return "ABBREV: Likely abbreviation"
    
    if len(term) <= 5 and term.replace('-', '').isalnum():
        return "ABBREV: Likely abbreviation"
    
    # Check if candidates exist
    if candidates:
        return "SYNONYM: Has potential matches"
    
    # Check patterns
    if 'cancer' in term_lower or 'carcinoma' in term_lower:
        return "MISSING: Cancer type not in DOID"
    
    if 'syndrome' in term_lower:
        return "MISSING: Syndrome not in DOID"
    
    return "UNKNOWN: No obvious match"


def analyze_diseases(
    ner_dir: str,
    index_path: str,
    top_n: int = 500,
    output_path: str = None
) -> Dict:
    """
    Analyze disease mentions and identify synonym candidates.
    
    Returns analysis summary.
    """
    
    # Load disease mentions
    diseases = load_disease_mentions(ner_dir)
    
    # Connect to FTS5 index
    print(f"\n📊 Checking against FTS5 index: {index_path}")
    conn = sqlite3.connect(index_path)
    conn.row_factory = sqlite3.Row
    
    # Analyze top N terms
    top_diseases = diseases.most_common(top_n)
    
    linked = []
    unlinked = []
    
    print(f"\n🔍 Analyzing top {len(top_diseases)} terms...")
    
    for term, count in top_diseases:
        match = check_fts5_match(conn, term)
        
        if match:
            linked.append({
                'term': term,
                'count': count,
                'doid_id': match[0],
                'doid_concept': match[1]
            })
        else:
            candidates = search_candidates(conn, term)
            classification = classify_unlinked_term(term, candidates)
            
            unlinked.append({
                'term': term,
                'count': count,
                'classification': classification,
                'candidates': candidates
            })
    
    conn.close()
    
    # Compute stats
    total_mentions_top = sum(d[1] for d in top_diseases)
    linked_mentions = sum(d['count'] for d in linked)
    unlinked_mentions = sum(d['count'] for d in unlinked)
    
    # Group unlinked by classification
    by_class = {}
    for item in unlinked:
        cls = item['classification'].split(':')[0]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(item)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"DISEASE LINKING ANALYSIS SUMMARY")
    print(f"{'='*70}")
    print(f"\nTop {top_n} terms coverage:")
    print(f"  Linked:   {len(linked):>5} terms ({linked_mentions:>7,} mentions, {100*linked_mentions/total_mentions_top:.1f}%)")
    print(f"  Unlinked: {len(unlinked):>5} terms ({unlinked_mentions:>7,} mentions, {100*unlinked_mentions/total_mentions_top:.1f}%)")
    
    print(f"\nUnlinked by category:")
    for cls, items in sorted(by_class.items(), key=lambda x: -sum(i['count'] for i in x[1])):
        cls_count = sum(i['count'] for i in items)
        print(f"  {cls}: {len(items)} terms ({cls_count:,} mentions)")
    
    # Print actionable items (ABBREV and SYNONYM)
    print(f"\n{'='*70}")
    print(f"ACTIONABLE: Abbreviations needing synonyms")
    print(f"{'='*70}")
    
    abbrev_items = by_class.get('ABBREV', [])
    for item in sorted(abbrev_items, key=lambda x: -x['count'])[:30]:
        print(f"\n  {item['term']} ({item['count']} mentions)")
        if item['candidates']:
            for doid_id, concept in item['candidates'][:3]:
                print(f"      → {concept} ({doid_id})")
    
    print(f"\n{'='*70}")
    print(f"ACTIONABLE: Terms with potential DOID matches")
    print(f"{'='*70}")
    
    synonym_items = by_class.get('SYNONYM', [])
    for item in sorted(synonym_items, key=lambda x: -x['count'])[:30]:
        print(f"\n  {item['term']} ({item['count']} mentions)")
        for doid_id, concept in item['candidates'][:3]:
            print(f"      → {concept} ({doid_id})")
    
    print(f"\n{'='*70}")
    print(f"SKIP: Generic terms (not diseases)")
    print(f"{'='*70}")
    
    skip_items = by_class.get('SKIP', [])
    for item in sorted(skip_items, key=lambda x: -x['count'])[:20]:
        print(f"  {item['count']:>5}  {item['term']}")
    
    # Generate output file if requested
    if output_path:
        output = {
            'summary': {
                'top_n': top_n,
                'linked_terms': len(linked),
                'linked_mentions': linked_mentions,
                'unlinked_terms': len(unlinked),
                'unlinked_mentions': unlinked_mentions,
            },
            'linked': linked,
            'unlinked': [
                {
                    'term': item['term'],
                    'count': item['count'],
                    'classification': item['classification'],
                    'candidates': [(str(c[0]), str(c[1])) for c in item['candidates']]
                }
                for item in unlinked
            ],
            'by_classification': {
                k: [
                    {
                        'term': i['term'], 
                        'count': i['count'], 
                        'candidates': [(str(c[0]), str(c[1])) for c in i['candidates']]
                    }
                    for i in v
                ] 
                for k, v in by_class.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n💾 Full analysis saved to: {output_path}")
    
    return {
        'linked': linked,
        'unlinked': unlinked,
        'by_class': by_class
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NER output to identify diseases needing synonym enrichment"
    )
    parser.add_argument(
        "--ner-dir", "-n",
        required=True,
        help="Path to NER output directory"
    )
    parser.add_argument(
        "--index", "-i",
        default="indices/cancer/doid_disease.db",
        help="Path to DOID FTS5 index"
    )
    parser.add_argument(
        "--top", "-t",
        type=int,
        default=500,
        help="Number of top terms to analyze (default: 500)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save full analysis JSON"
    )
    
    args = parser.parse_args()
    
    analyze_diseases(
        ner_dir=args.ner_dir,
        index_path=args.index,
        top_n=args.top,
        output_path=args.output
    )


if __name__ == "__main__":
    main()
