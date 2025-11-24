#!/usr/bin/env python3
"""
Unified Wikidata Taxonomy Enrichment Module

This module enriches taxonomy files with Wikidata IDs and aliases using multiple methods:
1. Direct Wikidata search with embedding-based matching
2. GENRE model for entity linking
3. Wikipedia redirect resolution

Usage:
    python wikidata_enricher.py --input taxonomies/energy/IRENA.tsv --output taxonomies/energy/IRENA_enriched.tsv
"""

import pandas as pd
import numpy as np
import requests
import time
import re
import os
import pickle
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "TaxonomyWikidataEnricher/1.0 (sirislab@sirisacademic.com)"

# Generic terms to exclude from aliases (can be extended per domain)
DEFAULT_EXCLUDE_ALIASES = {
    'gas', 'current', 'dish', 'turf', 'carbone', 'stubble',
    'element 92', 'element 94', 'star power', 'star energy',
    'black gold', 'top gas', 'oxygenates', 'limited energy',
    'gas production plant', 'stellar energy', 'stellar power'
}

# Wikidata entity types to exclude (too generic)
DISALLOWED_TYPES = {"scholarly article", "wikimedia disambiguation page"}


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class WikidataCandidate:
    """Represents a Wikidata entity candidate."""
    qid: str
    label: str
    aliases: List[str] = field(default_factory=list)
    description: str = ""
    entity_type: str = ""
    score: float = 0.0
    source: str = ""  # 'direct_search', 'genre', 'wikipedia'


# ============================================================================
# Wikidata API Functions
# ============================================================================

def wikidata_request(params: Dict[str, str], timeout: int = 30) -> Optional[Dict]:
    """Make a request to Wikidata API with error handling."""
    headers = {'User-Agent': USER_AGENT}
    url = 'https://www.wikidata.org/w/api.php'
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️ Request error: {e}")
        return None


def sparql_query(query: str, timeout: int = 30) -> Optional[Dict]:
    """Execute SPARQL query against Wikidata endpoint."""
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'application/sparql-results+json'
    }
    
    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={'query': query, 'format': 'json'},
            headers=headers,
            timeout=timeout
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"    ⚠️ SPARQL error: {e}")
        return None


def extract_qid(text: str) -> Optional[str]:
    """Extract Q-ID from Wikidata URL or text."""
    if not text or pd.isna(text):
        return None
    
    # Handle multiple QIDs separated by pipe
    if '|' in text:
        text = text.split('|')[0]
    
    # Extract QID (Q followed by digits)
    match = re.search(r'Q\d+', str(text))
    return match.group(0) if match else None


def get_wikidata_aliases_sparql(qid: str, language: str = "en") -> Dict[str, any]:
    """
    Retrieve label and aliases for a Wikidata entity via SPARQL.
    
    Returns:
        Dictionary with 'label' and 'aliases' keys
    """
    query = f"""
    SELECT ?label ?alias WHERE {{
      OPTIONAL {{ wd:{qid} rdfs:label ?label . FILTER(LANG(?label) = "{language}") }}
      OPTIONAL {{ wd:{qid} skos:altLabel ?alias . FILTER(LANG(?alias) = "{language}") }}
    }}
    """
    
    data = sparql_query(query)
    if not data:
        return {'label': None, 'aliases': []}
    
    results = data.get('results', {}).get('bindings', [])
    if not results:
        return {'label': None, 'aliases': []}
    
    # Extract label (same across all rows)
    label = results[0].get('label', {}).get('value') if results else None
    
    # Extract all unique aliases
    aliases = list(set(
        row.get('alias', {}).get('value')
        for row in results
        if 'alias' in row
    ))
    
    return {'label': label, 'aliases': aliases}


def search_wikidata_entities(query: str, language: str = "en") -> List[WikidataCandidate]:
    """
    Search Wikidata for entities matching the query.
    
    Returns:
        List of WikidataCandidate objects
    """
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'search': query,
        'language': language
    }
    
    res = wikidata_request(params)
    if not res or not res.get('success'):
        return []
    
    search_results = {}
    for r in res.get('search', []):
        search_results[r['id']] = WikidataCandidate(
            qid=r['id'],
            label=r.get('label', ''),
            aliases=r.get('aliases', []),
            description=r.get('description', ''),
            source='direct_search'
        )
    
    if not search_results:
        return []
    
    # Enrich with entity types (P31: instance of)
    ids = '|'.join(search_results.keys())
    entity_params = {
        'action': 'wbgetentities',
        'ids': ids,
        'format': 'json',
        'languages': language
    }
    
    entities = wikidata_request(entity_params)
    if entities and entities.get('success') and 'entities' in entities:
        for eid, ent in entities['entities'].items():
            type_id = ''
            if 'P31' in ent.get('claims', {}):
                claims = ent['claims']['P31']
                if claims and 'mainsnak' in claims[0]:
                    type_id = claims[0]['mainsnak'].get('datavalue', {}).get('value', {}).get('id', '')
            
            if type_id:
                # Fetch label for type
                type_res = wikidata_request({
                    'action': 'wbgetentities',
                    'ids': type_id,
                    'format': 'json',
                    'languages': language
                })
                if type_res:
                    type_label = type_res.get('entities', {}).get(type_id, {}).get('labels', {}).get(language, {}).get('value', '')
                    search_results[eid].entity_type = type_label
    
    # Filter out disallowed types
    return [
        candidate for candidate in search_results.values()
        if candidate.entity_type.lower() not in DISALLOWED_TYPES
    ]


def get_wikidata_label(qid: str, language: str = "en") -> Optional[str]:
    """Get the human-readable label for a Wikidata QID."""
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    
    try:
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        entities = data.get("entities", {})
        entity = entities.get(qid)
        if entity:
            return entity["labels"].get(language, {}).get("value")
    except Exception as e:
        print(f"    ⚠️ Error fetching label for {qid}: {e}")
    
    return None


# ============================================================================
# Wikipedia API Functions
# ============================================================================

def resolve_wikipedia_redirect(title: str, language: str = "en") -> str:
    """Follow Wikipedia redirect to get actual target page title."""
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": title,
        "redirects": 1,
        "format": "json"
    }
    
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        redirects = data.get("query", {}).get("redirects", [])
        if redirects:
            return redirects[0]["to"]
    except Exception as e:
        print(f"    ⚠️ Redirect error: {e}")
    
    return title


def get_wikipedia_qid(title: str, language: str = "en") -> Optional[str]:
    """Get Wikidata QID from Wikipedia page title."""
    url = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "pageprops",
        "format": "json",
        "titles": title
    }
    
    headers = {'User-Agent': USER_AGENT}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("pageprops", {}).get("wikibase_item")
    except Exception as e:
        print(f"    ⚠️ Wikipedia QID error: {e}")
    
    return None


# ============================================================================
# Embedding-based Matching
# ============================================================================

class SimilarityMatcher:
    """Handles embedding-based similarity matching."""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the sentence transformer model."""
        print(f"📦 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(
            model_name,
            tokenizer_kwargs={"clean_up_tokenization_spaces": False}
        )
    
    def build_text(self, concept: str, description: str = "", aliases: List[str] = None, 
                   entity_type: str = "") -> str:
        """Build rich text representation for embedding."""
        text = concept
        if description:
            text += f". Description: {description}"
        if aliases:
            text += f" Aliases: {'; '.join(aliases)}."
        if entity_type:
            text += f" Type: {entity_type}"
        return text
    
    def find_best_match(self, query_text: str, candidates: List[WikidataCandidate], 
                        threshold: float = 0.5) -> Tuple[Optional[WikidataCandidate], float]:
        """Find best matching candidate using cosine similarity."""
        if not candidates:
            return None, 0.0
        
        # Build candidate texts
        candidate_texts = [
            self.build_text(c.label, c.description, c.aliases, c.entity_type)
            for c in candidates
        ]
        
        # Encode
        sentences = [query_text] + candidate_texts
        embeddings = self.model.encode(sentences, show_progress_bar=False)
        
        # Calculate similarities
        scores = cosine_similarity([embeddings[0]], embeddings[1:]).flatten()
        
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        
        if best_score < threshold:
            return None, best_score
        
        best_candidate = candidates[best_idx]
        best_candidate.score = best_score
        return best_candidate, best_score


# ============================================================================
# GENRE Entity Linking
# ============================================================================

class GENRELinker:
    """Handles GENRE-based entity linking."""
    
    def __init__(self, model_name: str = "facebook/genre-linking-blink"):
        """Initialize GENRE model."""
        print(f"📦 Loading GENRE model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def link_entity(self, concept: str, description: str = "", 
                    num_beams: int = 5, num_return: int = 5) -> Optional[str]:
        """
        Generate Wikipedia title and retrieve Wikidata QID.
        
        Returns:
            Wikidata QID if found, None otherwise
        """
        # Build input sentence
        sentence = f"[START_ENT] {concept} [END_ENT]"
        if description:
            sentence += f" {description}"
        
        # Generate candidates
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                num_return_sequences=num_return,
                max_new_tokens=32
            )
        
        titles = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Try to get QID for each candidate
        for title in titles:
            title = title.strip()
            qid = get_wikipedia_qid(title)
            if qid:
                return qid
        
        return None


# ============================================================================
# Main Enrichment Class
# ============================================================================

class WikidataEnricher:
    """Main class for enriching taxonomy with Wikidata information."""
    
    def __init__(self, 
                 use_similarity: bool = True,
                 use_genre: bool = False,
                 similarity_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 similarity_threshold: float = 0.6,
                 exclude_aliases: Set[str] = None,
                 delay: float = 0.25):
        """
        Initialize the enricher.
        
        Args:
            use_similarity: Enable embedding-based similarity matching
            use_genre: Enable GENRE model for entity linking
            similarity_model: Model name for SimilarityMatcher
            similarity_threshold: Minimum similarity score for matching
            exclude_aliases: Set of generic terms to exclude from aliases
            delay: Delay between API requests (seconds)
        """
        self.use_similarity = use_similarity
        self.use_genre = use_genre
        self.similarity_threshold = similarity_threshold
        self.exclude_aliases = exclude_aliases or DEFAULT_EXCLUDE_ALIASES
        self.delay = delay
        
        # Initialize models
        self.similarity_matcher = SimilarityMatcher(similarity_model) if use_similarity else None
        self.genre_linker = GENRELinker() if use_genre else None
    
    def filter_aliases(self, aliases: List[str], existing_terms: Set[str], 
                      current_term: str) -> List[str]:
        """
        Filter aliases to remove generic terms and duplicates.
        
        Args:
            aliases: List of candidate aliases
            existing_terms: Set of terms already in taxonomy
            current_term: The current concept being processed
        
        Returns:
            Filtered list of valid aliases
        """
        filtered = []
        current_term_lower = current_term.lower().strip()
        
        for alias in aliases:
            alias_lower = alias.lower().strip()
            
            # Skip if in exclude list
            if alias_lower in self.exclude_aliases:
                continue
            
            # Skip if same as current term
            if alias_lower == current_term_lower:
                continue
            
            # Skip if already exists in taxonomy
            if alias_lower in existing_terms:
                continue
            
            filtered.append(alias)
        
        return filtered
    
    def find_wikidata_id(self, concept: str, description: str = "") -> Optional[str]:
        """
        Find Wikidata QID for a concept using multiple methods.
        
        Returns:
            Wikidata QID if found, None otherwise
        """
        # Method 1: Direct search with similarity matching
        if self.use_similarity:
            candidates = search_wikidata_entities(concept)
            if candidates:
                query_text = self.similarity_matcher.build_text(concept, description)
                match, score = self.similarity_matcher.find_best_match(
                    query_text, candidates, self.similarity_threshold
                )
                if match:
                    return match.qid
        
        # Method 2: GENRE model
        if self.use_genre:
            qid = self.genre_linker.link_entity(concept, description)
            if qid:
                return qid
        
        # Method 3: Wikipedia redirect
        real_title = resolve_wikipedia_redirect(concept)
        qid = get_wikipedia_qid(real_title)
        if qid:
            return qid
        
        return None
    
    def enrich_row(self, row: pd.Series, existing_terms: Set[str]) -> Dict:
        """Return dict of updates instead of modified Series"""
        updates = {}
        concept = row.get('concept', '')
        description = row.get('description', '')
        existing_qid = extract_qid(row.get('wikidata_id', ''))
        
        # If no QID, try to find one
        if not existing_qid:
            print(f"  🔍 Finding QID for '{concept}'...", end='')
            qid = self.find_wikidata_id(concept, description)
            if qid:
                updates['wikidata_id'] = f"https://www.wikidata.org/wiki/{qid}"
                print(f" ✓ {qid}")
            else:
                print(" ✗ Not found")
                return updates
        else:
            qid = existing_qid
        
        # Get aliases
        print(f"  📝 Fetching aliases for {qid}...", end='')
        result = get_wikidata_aliases_sparql(qid)
        
        valid_aliases = []
        if result['label'] and result['label'].lower() != concept.lower():
            if result['label'].lower() not in existing_terms:
                valid_aliases.append(result['label'])
        
        if result['aliases']:
            filtered = self.filter_aliases(result['aliases'], existing_terms, concept)
            valid_aliases.extend(filtered)
        
        if valid_aliases:
            valid_aliases = list(dict.fromkeys(valid_aliases))
            updates['wikidata_aliases'] = ' | '.join(valid_aliases)
            print(f" ✓ ({len(valid_aliases)} aliases)")
        else:
            print(" (no valid aliases)")
        
        return updates
    
    def enrich_taxonomy(self, input_path: str, output_path: str, 
                       resume: bool = True) -> pd.DataFrame:
        """
        Enrich entire taxonomy file with Wikidata information.
        
        Args:
            input_path: Path to input TSV file
            output_path: Path to output TSV file
            resume: Whether to resume from checkpoint
        
        Returns:
            Enriched DataFrame
        """
        print("=" * 80)
        print("Wikidata Taxonomy Enrichment")
        print("=" * 80)
        
        # Load taxonomy
        print(f"\n📖 Reading {input_path}...")
        df = pd.read_csv(input_path, sep='\t').fillna('')
        print(f"   Loaded {len(df)} rows")
        
        # Build set of existing terms
        print(f"\n📊 Building term index...")
        existing_terms = set()
        for concept in df['concept']:
            if pd.notna(concept) and concept:
                existing_terms.add(concept.lower().strip())
        print(f"   Found {len(existing_terms)} unique terms")
        
        # Initialize wikidata_aliases column if not present
        if 'wikidata_aliases' not in df.columns:
            df['wikidata_aliases'] = ''
        
        # Checkpoint handling
        checkpoint_path = output_path.replace('.tsv', '_checkpoint.pkl')
        start_idx = 0
        
        if resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                start_idx = checkpoint.get('last_index', 0)
                if 'df' in checkpoint:
                    df = checkpoint['df']
            print(f"\n📦 Resuming from checkpoint: row {start_idx}/{len(df)}")
        
        # Process rows
        print(f"\n🔄 Processing taxonomy entries...")
        processed = 0
        
        for idx in tqdm(range(start_idx, len(df)), desc="Enriching"):
            # Skip if already processed (has both QID and aliases)
            if df.at[idx, 'wikidata_id'] and df.at[idx, 'wikidata_aliases']:
                continue
            
            updates = self.enrich_row(df.iloc[idx], existing_terms)
            for key, value in updates.items():
                df.at[idx, key] = value
            processed += 1
            
            # Delay between requests
            if processed < len(df) - start_idx:
                time.sleep(self.delay)
            
            # Save checkpoint every 50 rows
            if (idx + 1) % 50 == 0:
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump({'last_index': idx + 1, 'df': df}, f)
        
        # Save final output
        print(f"\n💾 Saving enriched taxonomy to {output_path}...")
        df.to_csv(output_path, sep='\t', index=False)
        
        # Clean up checkpoint
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        # Statistics
        total_with_qid = (df['wikidata_id'] != '').sum()
        total_with_aliases = (df['wikidata_aliases'] != '').sum()
        
        print(f"\n✅ Done!")
        print(f"   Total rows: {len(df)}")
        print(f"   Rows with Wikidata IDs: {total_with_qid}")
        print(f"   Rows with aliases: {total_with_aliases}")
        
        # Show examples
        print(f"\n📋 Sample enriched entries:")
        sample = df[df['wikidata_aliases'] != ''].head(3)
        for _, row in sample.iterrows():
            print(f"\n   {row['concept']} ({row.get('id', 'N/A')})")
            print(f"   → {row['wikidata_id']}")
            print(f"   → {row['wikidata_aliases']}")
        
        return df


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enrich taxonomy with Wikidata IDs and aliases',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic enrichment with similarity matching
  python wikidata_enricher.py --input taxonomies/energy/IRENA.tsv --output taxonomies/energy/IRENA_enriched.tsv
  
  # With GENRE model (slower but more accurate)
  python wikidata_enricher.py --input taxonomy.tsv --output enriched.tsv --use-genre
  
  # Adjust similarity threshold
  python wikidata_enricher.py --input taxonomy.tsv --output enriched.tsv --threshold 0.7
        """
    )
    
    parser.add_argument(
        '--input',
        required=True,
        help='Input taxonomy TSV file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output enriched TSV file'
    )
    parser.add_argument(
        '--use-similarity',
        action='store_true',
        default=True,
        help='Use embedding-based similarity matching (default: True)'
    )
    parser.add_argument(
        '--no-similarity',
        dest='use_similarity',
        action='store_false',
        help='Disable similarity matching'
    )
    parser.add_argument(
        '--similarity-model',
        type=str,
        default='sentence-transformers/all-MiniLM-L6-v2',
        help='Sentence transformer model for similarity matching'
    )
    parser.add_argument(
        '--use-genre',
        action='store_true',
        default=False,
        help='Use GENRE model for entity linking (slower)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Similarity threshold for matching (0.0-1.0, default: 0.6)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.25,
        help='Delay between API requests in seconds (default: 0.25)'
    )
    parser.add_argument(
        '--no-resume',
        dest='resume',
        action='store_false',
        default=True,
        help='Start from scratch instead of resuming from checkpoint'
    )
    parser.add_argument(
        '--exclude-aliases',
        type=str,
        help='Comma-separated list of additional aliases to exclude'
    )
    
    args = parser.parse_args()
    
    # Build exclude set
    exclude_aliases = DEFAULT_EXCLUDE_ALIASES.copy()
    if args.exclude_aliases:
        custom_excludes = {a.strip().lower() for a in args.exclude_aliases.split(',')}
        exclude_aliases.update(custom_excludes)
    
    # Create enricher
    enricher = WikidataEnricher(
        use_similarity=args.use_similarity,
        use_genre=args.use_genre,
        similarity_model=args.similarity_model,
        similarity_threshold=args.threshold,
        exclude_aliases=exclude_aliases,
        delay=args.delay
    )
    
    # Run enrichment
    enricher.enrich_taxonomy(args.input, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
