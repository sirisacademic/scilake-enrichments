"""
Gazetteer-based entity linking using FlashText for fast exact matching.
Supports case-insensitive matching with intelligent acronym handling.
"""

import os
import pandas as pd
from flashtext import KeywordProcessor
from typing import List, Dict, Any, Optional, Set
from pathlib import Path


class GazetteerLinker:
    def __init__(
        self, 
        taxonomy_path: str, 
        taxonomy_source: str = None, 
        model_name: str = None, 
        default_type: str = None, 
        domain: str = None,
        min_term_length: int = 2,
        blocked_terms: Optional[Set[str]] = None,
        logger=None
    ):
        """
        Initialize gazetteer linker with taxonomy.
        
        Args:
            taxonomy_path: Path to taxonomy TSV file
            taxonomy_source: Source name for linking metadata (auto-detected if None)
            model_name: Model name for entity metadata (auto-generated if None)
            default_type: Default entity type when taxonomy type is empty
            domain: Domain name (e.g., 'ccam', 'energy', 'neuro')
            min_term_length: Minimum character length for matches (default: 2)
            blocked_terms: Set of lowercase terms to always reject
            logger: Optional logger instance
        """
        # Resolve relative to project root
        if not os.path.isabs(taxonomy_path):
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        self.taxonomy_df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        # Auto-detect or use provided values
        self.taxonomy_source = taxonomy_source or self._detect_source(taxonomy_path)
        self.model_name = model_name or f"{self.taxonomy_source}-Gazetteer"
        self.default_type = default_type
        self.domain = domain
        self.min_term_length = min_term_length
        self.blocked_terms = blocked_terms or set()
        self.logger = logger
        
        # Track which terms contain acronyms (for case validation)
        self._acronym_patterns = {}  # concept -> list of (word_index, acronym_word)
        
        self._build_index()
    
    def _is_acronym_word(self, word: str) -> bool:
        """
        Check if a word is an acronym (all letters are uppercase, at least 2 letters).
        
        Handles hyphenated acronyms like "C-V2X", "C-ITS".
        """
        # Extract only alphabetic characters
        letters = [c for c in word if c.isalpha()]
        return len(letters) >= 2 and all(c.isupper() for c in letters)
    
    def _find_acronym_positions(self, text: str) -> List[tuple]:
        """
        Find positions of acronym words in a phrase.
        
        Returns:
            List of (word_index, acronym_word) tuples
        """
        words = text.split()
        acronym_positions = []
        
        for i, word in enumerate(words):
            if self._is_acronym_word(word):
                acronym_positions.append((i, word))
        
        return acronym_positions
    
    def _should_accept_match(self, matched_text: str, concept: str) -> bool:
        """
        Check if the matched text has correct case for acronym parts.
        
        Acronyms in the taxonomy concept must appear uppercase in the matched text.
        Non-acronym parts can be any case (case-insensitive).
        
        Examples:
            - "ITS user" matches "ITS user" → True (acronym preserved)
            - "its user" matches "ITS user" → False (acronym not preserved)
            - "vehicle" matches "Vehicle" → True (not an acronym)
            - "C-V2X" matches "C-V2X" → True (acronym preserved)
            - "c-v2x" matches "C-V2X" → False (acronym not preserved)
        """
        # Get acronym positions for this concept
        acronym_positions = self._acronym_patterns.get(concept.lower(), [])
        
        if not acronym_positions:
            # No acronyms in concept - accept any case
            return True
        
        # Split both into words for comparison
        matched_words = matched_text.split()
        concept_words = concept.split()
        
        # If word counts don't match, we can't validate properly - accept it
        if len(matched_words) != len(concept_words):
            return True
        
        # Check each acronym position
        for word_idx, acronym_word in acronym_positions:
            if word_idx >= len(matched_words):
                continue
            
            matched_word = matched_words[word_idx]
            
            # Check if the matched word preserves the uppercase letters
            if not self._is_acronym_word(matched_word):
                # Matched word is not uppercase - reject
                if self.logger:
                    self.logger.debug(
                        f"Rejecting '{matched_text}' → '{concept}': "
                        f"acronym '{acronym_word}' not preserved (got '{matched_word}')"
                    )
                return False
        
        return True
    
    def _build_index(self):
        """Build FlashText index with all terms and aliases."""
        
        # Aliases
        aliases_cols = ["synonyms", "wikidata_aliases"]
        existing = [c for c in aliases_cols if c in self.taxonomy_df.columns]
        self.taxonomy_df["aliases"] = (
            self.taxonomy_df[existing]
            .fillna("")
            .apply(lambda row: " | ".join([v for v in row if v]), axis=1)
        )
        
        for _, row in self.taxonomy_df.iterrows():
            metadata = {
                'taxonomy_id': str(row['id']),
                'concept': row['concept'],
                'wikidata_id': row.get('wikidata_id', ''),
                'type': row.get('type', '')
            }
            
            # Track acronym positions for primary concept
            concept = row['concept']
            acronym_positions = self._find_acronym_positions(concept)
            if acronym_positions:
                self._acronym_patterns[concept.lower()] = acronym_positions
            
            # Add primary concept
            self.keyword_processor.add_keyword(concept, metadata)
            
            # Add aliases
            if pd.notna(row.get('aliases')):
                for alias in row['aliases'].split(' | '):
                    alias = alias.strip()
                    if alias:
                        # Track acronym positions for alias too
                        alias_acronym_positions = self._find_acronym_positions(alias)
                        if alias_acronym_positions:
                            self._acronym_patterns[alias.lower()] = alias_acronym_positions
                        
                        # Create metadata copy with the alias as matched form
                        alias_metadata = metadata.copy()
                        alias_metadata['matched_form'] = alias
                        self.keyword_processor.add_keyword(alias, alias_metadata)
        
        if self.logger:
            acronym_count = len(self._acronym_patterns)
            self.logger.info(
                f"✅ Gazetteer index built: {len(self.taxonomy_df)} concepts, "
                f"{acronym_count} terms with acronyms"
            )
    
    def _detect_source(self, taxonomy_path: Path) -> str:
        """Auto-detect taxonomy source from filename."""
        filename = Path(taxonomy_path).stem
        
        if 'IRENA' in filename:
            return 'IRENA'
        elif 'Neuroscience' in filename or 'UBERON' in filename:
            return 'OPENMINDS-UBERON'
        elif 'Vessel' in filename or 'Maritime' in filename:
            return 'Maritime-Ontology'
        elif 'CCAM' in filename:
            return 'SINFONICA-FAME'
        else:
            return filename
    
    def extract_entities(
        self, 
        text: str, 
        section_id: str, 
        domain: str = None
    ) -> List[Dict[str, Any]]:
        """
        Extract gazetteer matches from text with acronym case validation.
        
        Args:
            text: Input text to search
            section_id: Section identifier for metadata
            domain: Domain name (uses self.domain if not provided)
            
        Returns:
            List of entity dicts with linking information
        """
        domain = domain or self.domain
        matches = self.keyword_processor.extract_keywords(text, span_info=True)
        
        entities = []
        for metadata, start, end in matches:
            matched_text = text[start:end]
            concept = metadata['concept']
            
            # Get the form that was actually matched (could be alias)
            matched_form = metadata.get('matched_form', concept)

            # FIX: Discard broken FlashText matches (offset bug with special chars)
            if matched_text.lower() != matched_form.lower():
                if self.logger:
                    self.logger.debug(
                        f"Skipping broken FlashText match: expected '{matched_form}' "
                        f"but got '{matched_text}' at {start}-{end}"
                    )
                continue

            # Apply filters
            
            # 1. Minimum length filter
            if len(matched_text) < self.min_term_length:
                if self.logger:
                    self.logger.debug(f"Skipping '{matched_text}': below min length {self.min_term_length}")
                continue
            
            # 2. Blocked terms filter
            if matched_text.lower() in self.blocked_terms:
                if self.logger:
                    self.logger.debug(f"Skipping '{matched_text}': in blocked terms")
                continue
            
            # 3. Acronym case validation
            if not self._should_accept_match(matched_text, matched_form):
                continue
            
            entity = {
                "entity": self._map_type(metadata['type'], domain),
                "text": matched_text,
                "score": 1.0,  # Exact match
                "start": start,
                "end": end,
                "model": self.model_name,
                "domain": domain,
                "section_id": section_id,
                "linking": self._create_linking(metadata)
            }
            entities.append(entity)
        
        return entities
    
    def _map_type(self, taxonomy_type: str, domain: str = None) -> str:
        """Map taxonomy type to NER entity type based on domain."""
        domain = domain or self.domain
        
        # Handle empty types
        if not taxonomy_type or taxonomy_type.strip() == "":
            if self.default_type:
                return self.default_type
            defaults = {
                'neuro': 'UBERONParcellation',
                'maritime': 'vesselType',
                'energy': 'energytype',
                'ccam': 'General terms related to CCAM'
            }
            return defaults.get(domain, 'Unknown')
        
        # Domain-specific mappings
        if domain == 'energy':
            energy_map = {
                'Renewables': 'energytype',
                'Non-renewable': 'energytype',
                'Storage': 'energystorage'
            }
            return energy_map.get(taxonomy_type, 'energytype')
        
        elif domain == 'maritime':
            return 'vesselType'
        
        else:
            return taxonomy_type
    
    def _create_linking(self, metadata: Dict) -> List[Dict[str, str]]:
        """Create linking structure."""
        linking = [{
            "source": self.taxonomy_source,
            "id": metadata['taxonomy_id'],
            "name": metadata['concept']
        }]
        
        if metadata['wikidata_id']:
            wd_id = metadata['wikidata_id'].split('/')[-1]
            linking.append({
                "source": "Wikidata",
                "id": wd_id,
                "name": metadata['concept']
            })
        
        return linking


# --------------------------------------------------
# CLI for testing
# --------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Gazetteer Linker")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy TSV")
    parser.add_argument("--text", required=True, help="Text to search")
    parser.add_argument("--domain", default="ccam", help="Domain name")
    args = parser.parse_args()
    
    linker = GazetteerLinker(
        taxonomy_path=args.taxonomy,
        domain=args.domain
    )
    
    entities = linker.extract_entities(
        text=args.text,
        section_id="test_section",
        domain=args.domain
    )
    
    print(f"\nFound {len(entities)} entities:")
    for e in entities:
        print(f"  '{e['text']}' [{e['start']}:{e['end']}] → {e['linking'][0]['name']}")
