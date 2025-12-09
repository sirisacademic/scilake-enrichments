#!/usr/bin/env python3
"""
FTS5Linker: Entity linking using SQLite FTS5 indices.

Provides fast exact matching against large vocabularies (millions of entries)
without loading everything into memory.

Features:
- Case-insensitive matching on concept and synonyms
- Text normalization (Greek letters, spacing, plurals)
- Fast exact matching using SQL indices
- Frequency-based disambiguation when multiple entities match

Usage:
    linker = FTS5Linker(
        index_path="indices/cancer/ncbi_gene.db",
        taxonomy_source="NCBI_Gene"
    )
    
    result = linker.link_entity("TP53")
    # Returns: [{"source": "NCBI_Gene", "id": "NCBI:7157", "name": "TP53", "score": 1.0}]
    
    # Also matches normalized variants:
    result = linker.link_entity("ifn-γ")  # Matches "ifng" via Greek normalization
    result = linker.link_entity("erk1 / 2")  # Matches "erk1/2" via spacing normalization
"""

import sqlite3
import re
from typing import List, Dict, Optional, Set
from pathlib import Path


# =============================================================================
# Text Normalization
# =============================================================================

# Greek letter mappings (lowercase)
GREEK_TO_LATIN = {
    'α': 'a', 'β': 'b', 'γ': 'g', 'δ': 'd', 'ε': 'e',
    'ζ': 'z', 'η': 'e', 'θ': 'th', 'ι': 'i', 'κ': 'k',
    'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x', 'ο': 'o',
    'π': 'p', 'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't',
    'υ': 'u', 'φ': 'ph', 'χ': 'ch', 'ψ': 'ps', 'ω': 'o',
    # Uppercase
    'Α': 'A', 'Β': 'B', 'Γ': 'G', 'Δ': 'D', 'Ε': 'E',
    'Ζ': 'Z', 'Η': 'E', 'Θ': 'Th', 'Ι': 'I', 'Κ': 'K',
    'Λ': 'L', 'Μ': 'M', 'Ν': 'N', 'Ξ': 'X', 'Ο': 'O',
    'Π': 'P', 'Ρ': 'R', 'Σ': 'S', 'Τ': 'T', 'Υ': 'U',
    'Φ': 'Ph', 'Χ': 'Ch', 'Ψ': 'Ps', 'Ω': 'O',
}


def normalize_greek(text: str) -> Optional[str]:
    """
    Replace Greek letters with Latin equivalents.
    Returns None if no Greek letters found.
    
    Examples:
        "ifn-γ" -> "ifn-g"
        "tgf-β" -> "tgf-b"
        "nf-κb" -> "nf-kb"
    """
    has_greek = any(c in GREEK_TO_LATIN for c in text)
    if not has_greek:
        return None
    
    result = []
    for char in text:
        result.append(GREEK_TO_LATIN.get(char, char))
    return ''.join(result)


def normalize_greek_aggressive(text: str) -> Optional[str]:
    """
    Replace Greek letters AND remove hyphens before them.
    Returns None if no Greek letters found.
    
    Examples:
        "ifn-γ" -> "ifng"
        "tgf-β" -> "tgfb"
    """
    has_greek = any(c in GREEK_TO_LATIN for c in text)
    if not has_greek:
        return None
    
    result = []
    skip_next_hyphen = False
    
    for i, char in enumerate(text):
        if char == '-' and i + 1 < len(text) and text[i + 1] in GREEK_TO_LATIN:
            # Skip hyphen before Greek letter
            continue
        result.append(GREEK_TO_LATIN.get(char, char))
    
    return ''.join(result)


def normalize_spacing(text: str) -> Optional[str]:
    """
    Normalize spacing around punctuation.
    Returns None if no changes made.
    
    Examples:
        "erk1 / 2" -> "erk1/2"
        "il - 6" -> "il-6"
    """
    # Remove spaces around slashes and hyphens
    normalized = re.sub(r'\s*/\s*', '/', text)
    normalized = re.sub(r'\s*-\s*', '-', normalized)
    
    if normalized == text:
        return None
    return normalized


def depluralize(text: str) -> Optional[str]:
    """
    Simple depluralization for common patterns.
    Returns None if no changes made.
    
    Examples:
        "cytokines" -> "cytokine"
        "receptors" -> "receptor"
    """
    if len(text) < 4:
        return None
    
    # Common plural endings
    if text.endswith('ies'):
        return text[:-3] + 'y'
    elif text.endswith('es') and not text.endswith('ases'):
        return text[:-2]
    elif text.endswith('s') and not text.endswith('ss'):
        return text[:-1]
    
    return None


def generate_variants(text: str) -> List[str]:
    """
    Generate normalized variants of text for matching.
    
    Returns list of unique variants (not including original).
    
    Order of variants (tried first to last):
    1. Greek normalized (ifn-γ -> ifn-g)
    2. Greek aggressive (ifn-γ -> ifng)
    3. Spacing normalized (erk1 / 2 -> erk1/2)
    4. Depluralized (chemokines -> chemokine)
    5. Combinations (Greek + spacing, Greek + depluralize, etc.)
    """
    variants = []
    seen = {text.lower()}
    
    def add_variant(v):
        if v and v.lower() not in seen:
            variants.append(v)
            seen.add(v.lower())
    
    # Single normalizations
    greek = normalize_greek(text)
    add_variant(greek)
    
    greek_agg = normalize_greek_aggressive(text)
    add_variant(greek_agg)
    
    spacing = normalize_spacing(text)
    add_variant(spacing)
    
    deplural = depluralize(text)
    add_variant(deplural)
    
    # Combinations
    if greek:
        add_variant(normalize_spacing(greek))
        add_variant(depluralize(greek))
    
    if greek_agg:
        add_variant(normalize_spacing(greek_agg))
        add_variant(depluralize(greek_agg))
    
    if spacing:
        add_variant(normalize_greek(spacing))
        add_variant(normalize_greek_aggressive(spacing))
        add_variant(depluralize(spacing))
    
    if deplural:
        add_variant(normalize_greek(deplural))
        add_variant(normalize_greek_aggressive(deplural))
        add_variant(normalize_spacing(deplural))
    
    return variants


# =============================================================================
# FTS5Linker Class
# =============================================================================

class FTS5Linker:
    """
    Entity linker using SQLite FTS5 for fast exact matching.
    
    Supports matching against:
    - Primary concept name
    - Synonyms (pipe-separated in synonyms column)
    - Normalized variants (Greek letters, spacing, plurals)
    
    All matching is case-insensitive.
    Uses frequency for disambiguation when multiple entities match.
    """
    
    def __init__(
        self,
        index_path: str,
        taxonomy_source: str,
        logger=None,
        enable_normalization: bool = True
    ):
        """
        Initialize FTS5 linker.
        
        Args:
            index_path: Path to SQLite FTS5 database
            taxonomy_source: Source name for linking output (e.g., "NCBI_Gene")
            logger: Optional logger instance
            enable_normalization: Enable text normalization (Greek, spacing, plurals)
        """
        self.index_path = Path(index_path)
        self.taxonomy_source = taxonomy_source
        self.logger = logger
        self.enable_normalization = enable_normalization
        
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        # Open database connection
        self.conn = sqlite3.connect(str(self.index_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Verify database structure
        self._verify_database()
        
        if self.logger:
            cursor = self.conn.execute("SELECT COUNT(*) FROM entities")
            count = cursor.fetchone()[0]
            self.logger.info(f"✅ FTS5Linker loaded: {index_path} ({count:,} entries)")
    
    def _verify_database(self):
        """Verify database has required tables."""
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        
        required = {'entities', 'entities_fts'}
        missing = required - tables
        
        if missing:
            raise ValueError(f"Database missing required tables: {missing}")
    
    def link_entity(
        self,
        entity_text: str,
        context: str = None  # Not used, but kept for interface compatibility
    ) -> Optional[List[Dict]]:
        """
        Link entity text to vocabulary entry.
        
        Matching strategy:
        1. Try original text (concept, then synonyms)
        2. Try normalized variants (Greek, spacing, plurals)
        
        When multiple entities match, returns the one with highest frequency.
        
        Args:
            entity_text: Entity mention to link
            context: Not used (for interface compatibility)
            
        Returns:
            List with linking dict, or None if no match
        """
        if not entity_text or not entity_text.strip():
            return None
        
        entity_text = entity_text.strip()
        
        # Try original text
        result = self._try_match(entity_text)
        if result:
            return result
        
        # Try normalized variants
        if self.enable_normalization:
            variants = generate_variants(entity_text)
            for variant in variants:
                result = self._try_match(variant)
                if result:
                    if self.logger:
                        self.logger.debug(
                            f"FTS5: '{entity_text}' matched via variant '{variant}'"
                        )
                    return result
        
        # No match found
        if self.logger:
            self.logger.debug(f"FTS5: No match for '{entity_text}'")
        
        return None
    
    def _try_match(self, text: str) -> Optional[List[Dict]]:
        """
        Try to match text against concept and synonyms.
        
        Args:
            text: Text to match
            
        Returns:
            Formatted result or None
        """
        text_lower = text.lower()
        
        # Strategy 1: Exact match on concept
        result = self._match_concept(text_lower)
        if result:
            return self._format_result(result, text)
        
        # Strategy 2: Exact match on synonym
        result = self._match_synonym(text_lower)
        if result:
            return self._format_result(result, text)
        
        return None
    
    def _match_concept(self, entity_lower: str) -> Optional[sqlite3.Row]:
        """
        Try exact match on concept column (uses idx_concept index).
        Returns entity with highest frequency when multiple match.
        """
        cursor = self.conn.execute(
            """
            SELECT id, concept, synonyms, description, type, frequency
            FROM entities
            WHERE concept = ? COLLATE NOCASE
            ORDER BY frequency DESC
            LIMIT 1
            """,
            (entity_lower,)
        )
        return cursor.fetchone()
    
    def _match_synonym(self, entity_lower: str) -> Optional[sqlite3.Row]:
        """
        Try exact match on synonym_lookup table (fast indexed lookup).
        Returns the entity with highest frequency when multiple match.
        """
        cursor = self.conn.execute(
            """
            SELECT e.id, e.concept, e.synonyms, e.description, e.type, e.frequency
            FROM synonym_lookup sl
            JOIN entities e ON sl.entity_id = e.id
            WHERE sl.synonym = ?
            ORDER BY e.frequency DESC
            LIMIT 1
            """,
            (entity_lower,)
        )
        return cursor.fetchone()
    
    def _format_result(
        self,
        row: sqlite3.Row,
        original_text: str
    ) -> List[Dict]:
        """Format database row as linking result."""
        result = {
            "source": self.taxonomy_source,
            "id": row["id"],
            "name": row["concept"],
            "score": 1.0  # Exact match
        }
        
        # Include frequency if available (for debugging/analysis)
        try:
            freq = row["frequency"]
            if freq and int(freq) > 0:
                result["frequency"] = int(freq)
        except (KeyError, TypeError, ValueError):
            pass
        
        return [result]
    
    def link_entities_batch(
        self,
        entities: List[str]
    ) -> Dict[str, Optional[List[Dict]]]:
        """
        Link multiple entities (convenience method).
        
        Args:
            entities: List of entity texts
            
        Returns:
            Dict mapping entity text to linking result (or None)
        """
        results = {}
        for entity in entities:
            results[entity] = self.link_entity(entity)
        return results
    
    def get_entry_by_id(self, entity_id: str) -> Optional[Dict]:
        """
        Look up entry by ID (for debugging/inspection).
        
        Args:
            entity_id: Entity ID (e.g., "NCBI:7157")
            
        Returns:
            Full entry dict or None
        """
        cursor = self.conn.execute(
            """
            SELECT id, concept, synonyms, description, type, taxonomy_source, frequency
            FROM entities
            WHERE id = ?
            """,
            (entity_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def get_all_matches(self, entity_text: str, limit: int = 10) -> List[Dict]:
        """
        Get all matching entities for a term (for debugging/analysis).
        
        Useful to see which entities would match and their frequencies.
        
        Args:
            entity_text: Entity mention to look up
            limit: Maximum results to return
            
        Returns:
            List of matching entities with their frequencies
        """
        entity_lower = entity_text.lower().strip()
        
        results = []
        
        # Check synonym matches
        cursor = self.conn.execute(
            """
            SELECT e.id, e.concept, e.frequency
            FROM synonym_lookup sl
            JOIN entities e ON sl.entity_id = e.id
            WHERE sl.synonym = ?
            ORDER BY e.frequency DESC
            LIMIT ?
            """,
            (entity_lower, limit)
        )
        
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "concept": row["concept"],
                "frequency": row["frequency"] or 0,
                "match_type": "synonym"
            })
        
        # Also check concept matches
        cursor = self.conn.execute(
            """
            SELECT id, concept, frequency
            FROM entities
            WHERE concept = ? COLLATE NOCASE
            ORDER BY frequency DESC
            LIMIT ?
            """,
            (entity_lower, limit)
        )
        
        for row in cursor.fetchall():
            # Avoid duplicates
            if not any(r["id"] == row["id"] for r in results):
                results.append({
                    "id": row["id"],
                    "concept": row["concept"],
                    "frequency": row["frequency"] or 0,
                    "match_type": "concept"
                })
        
        # Sort by frequency
        results.sort(key=lambda x: x["frequency"], reverse=True)
        
        return results[:limit]
    
    def search_fts(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Full-text search (for debugging/exploration).
        
        This uses FTS5 MATCH which does tokenized search,
        not the exact matching used by link_entity().
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            List of matching entries
        """
        cursor = self.conn.execute(
            """
            SELECT e.id, e.concept, e.synonyms, e.description, e.type, e.frequency
            FROM entities e
            JOIN entities_fts fts ON e.rowid = fts.rowid
            WHERE entities_fts MATCH ?
            ORDER BY e.frequency DESC
            LIMIT ?
            """,
            (query, limit)
        )
        
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __repr__(self):
        return f"FTS5Linker(index={self.index_path.name}, source={self.taxonomy_source})"


# =============================================================================
# FTS5LinkerManager Class
# =============================================================================

class FTS5LinkerManager:
    """
    Manager for multiple FTS5 linkers (one per entity type).
    
    Usage:
        manager = FTS5LinkerManager(config, logger)
        linking = manager.link_entity("TP53", "gene")
    """
    
    def __init__(
        self,
        config: Dict[str, Dict],
        logger=None
    ):
        """
        Initialize manager with config.
        
        Args:
            config: Dict mapping entity type to linker config
                {
                    "gene": {"index_path": "...", "taxonomy_source": "NCBI_Gene"},
                    "species": {"index_path": "...", "taxonomy_source": "NCBI_Taxonomy"},
                    ...
                }
            logger: Optional logger
        """
        self.logger = logger
        self.linkers: Dict[str, FTS5Linker] = {}
        
        for entity_type, linker_config in config.items():
            entity_type_lower = entity_type.lower()
            try:
                self.linkers[entity_type_lower] = FTS5Linker(
                    index_path=linker_config["index_path"],
                    taxonomy_source=linker_config["taxonomy_source"],
                    logger=logger
                )
            except Exception as e:
                if logger:
                    logger.error(f"Failed to load FTS5 index for {entity_type}: {e}")
        
        if logger:
            logger.info(f"FTS5LinkerManager ready: {list(self.linkers.keys())}")
    
    def link_entity(
        self,
        entity_text: str,
        entity_type: str
    ) -> Optional[List[Dict]]:
        """
        Link entity using appropriate linker for entity type.
        
        Args:
            entity_text: Entity mention
            entity_type: Entity type (e.g., "gene", "species")
            
        Returns:
            Linking result or None
        """
        entity_type_lower = entity_type.lower()
        
        linker = self.linkers.get(entity_type_lower)
        if not linker:
            if self.logger:
                self.logger.debug(f"No FTS5 linker for entity type: {entity_type}")
            return None
        
        return linker.link_entity(entity_text)
    
    def has_linker(self, entity_type: str) -> bool:
        """Check if linker exists for entity type."""
        return entity_type.lower() in self.linkers
    
    def close_all(self):
        """Close all linker connections."""
        for linker in self.linkers.values():
            linker.close()
        self.linkers.clear()


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FTS5Linker")
    parser.add_argument("--index", "-i", required=True, help="Path to FTS5 index")
    parser.add_argument("--source", "-s", required=True, help="Taxonomy source name")
    parser.add_argument("--query", "-q", required=True, help="Entity text to link")
    parser.add_argument("--no-normalize", action="store_true", help="Disable normalization")
    parser.add_argument("--show-variants", action="store_true", help="Show generated variants")
    parser.add_argument("--show-all-matches", action="store_true", help="Show all matching entities")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    
    args = parser.parse_args()
    
    # Show variants if requested
    if args.show_variants:
        print(f"Variants for '{args.query}':")
        variants = generate_variants(args.query)
        if variants:
            for v in variants:
                print(f"   → {v}")
        else:
            print("   (no variants generated)")
        print()
    
    linker = FTS5Linker(
        index_path=args.index,
        taxonomy_source=args.source,
        enable_normalization=not args.no_normalize
    )
    
    print(f"Linking: '{args.query}'")
    result = linker.link_entity(args.query)
    
    if result:
        print(f"✅ Match found:")
        for r in result:
            print(f"   {r}")
    else:
        print("❌ No match")
    
    # Show all matches if requested
    if args.show_all_matches:
        print(f"\nAll matching entities for '{args.query}':")
        all_matches = linker.get_all_matches(args.query)
        if all_matches:
            for m in all_matches:
                print(f"   {m['concept']} ({m['id']}) - freq: {m['frequency']}, type: {m['match_type']}")
        else:
            print("   (no matches)")
    
    if args.debug:
        print(f"\nFTS5 search results for '{args.query}':")
        fts_results = linker.search_fts(args.query, limit=5)
        for r in fts_results:
            print(f"   {r['concept']} ({r['id']}) - freq: {r.get('frequency', 0)}")
    
    linker.close()
