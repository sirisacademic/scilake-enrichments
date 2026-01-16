#!/usr/bin/env python3
"""
Type Matching for Entity Linking

Validates that linked entities have matching types between NER output and taxonomy.
Helps filter out incorrect linkings where the text matches but the type doesn't.

Example:
    NER detects "Paris" as type "species" (incorrect)
    Linker matches to concept with type "Location" 
    Type match: species ≠ location → REJECT linking

Usage:
    from src.type_matching import TypeMatcher
    
    matcher = TypeMatcher(domain_conf, taxonomy_path, logger)
    
    # Check single entity
    is_match = matcher.check_type_match(
        ner_type="gene",
        taxonomy_id="NCBI:7157",
        taxonomy_type="Gene"
    )
    
    # Filter linking result
    filtered_linking = matcher.filter_linking_by_type(
        entity_type="gene",
        linking_result=[{"id": "NCBI:7157", "name": "TP53", ...}]
    )
"""

import pandas as pd
from typing import Dict, List, Optional, Set, Union, Tuple
from pathlib import Path


class TypeMatcher:
    """
    Handles type matching between NER entity types and taxonomy concept types.
    
    Supports:
    - Domain-specific type mappings (configurable)
    - Case-insensitive matching
    - Multiple valid NER types per taxonomy type
    - Taxonomy type loading from TSV files
    """
    
    def __init__(
        self,
        domain_conf: dict,
        taxonomy_path: Optional[str] = None,
        logger=None
    ):
        """
        Initialize TypeMatcher.
        
        Args:
            domain_conf: Domain configuration from DOMAIN_MODELS
            taxonomy_path: Path to taxonomy TSV (for loading concept types)
            logger: Optional logger instance
        """
        self.logger = logger
        self.domain_conf = domain_conf
        
        # Load configuration
        self.enabled = domain_conf.get("enforce_type_match", True)
        self.type_mappings = domain_conf.get("type_mappings", {})
        self.type_column = domain_conf.get("taxonomy_type_column", "type")
        
        # Normalize type mappings (lowercase keys)
        self.type_mappings = {
            k.lower(): self._normalize_mapping_value(v)
            for k, v in self.type_mappings.items()
        }
        
        # Load taxonomy types if path provided
        self.taxonomy_types: Dict[str, str] = {}
        if taxonomy_path and Path(taxonomy_path).exists():
            self._load_taxonomy_types(taxonomy_path)
        
        if self.logger and self.enabled:
            self.logger.info(f"✅ TypeMatcher initialized: {len(self.type_mappings)} type mappings")
            if self.taxonomy_types:
                self.logger.info(f"   Loaded types for {len(self.taxonomy_types)} taxonomy concepts")
    
    def _normalize_mapping_value(self, value: Union[str, List[str]]) -> Set[str]:
        """
        Normalize mapping value to a set of lowercase strings.
        
        Supports:
        - Single string: "gene" → {"gene"}
        - List of strings: ["gene", "protein"] → {"gene", "protein"}
        """
        if isinstance(value, str):
            return {value.lower()}
        elif isinstance(value, (list, tuple, set)):
            return {str(v).lower() for v in value}
        else:
            return {str(value).lower()}
    
    def _load_taxonomy_types(self, taxonomy_path: str):
        """
        Load concept ID → type mapping from taxonomy TSV.
        
        Expected TSV columns:
        - id: Concept identifier
        - type (or custom column via taxonomy_type_column): Concept type
        """
        try:
            df = pd.read_csv(taxonomy_path, sep='\t', dtype=str).fillna("")
            
            if 'id' not in df.columns:
                if self.logger:
                    self.logger.warning(f"⚠️ Taxonomy missing 'id' column: {taxonomy_path}")
                return
            
            if self.type_column not in df.columns:
                if self.logger:
                    self.logger.warning(
                        f"⚠️ Taxonomy missing type column '{self.type_column}': {taxonomy_path}"
                    )
                return
            
            # Build mapping: id → type
            for _, row in df.iterrows():
                tax_id = str(row['id']).strip()
                tax_type = str(row[self.type_column]).strip()
                if tax_id and tax_type:
                    self.taxonomy_types[tax_id] = tax_type
            
            if self.logger:
                self.logger.debug(f"Loaded {len(self.taxonomy_types)} concept types from {taxonomy_path}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Failed to load taxonomy types from {taxonomy_path}: {e}")
    
    def get_expected_ner_types(self, taxonomy_type: str) -> Set[str]:
        """
        Get the expected NER type(s) for a taxonomy type.
        
        Uses type_mappings if defined, otherwise falls back to lowercase taxonomy type.
        
        Args:
            taxonomy_type: Type from taxonomy (e.g., "Gene", "Storage")
            
        Returns:
            Set of valid NER types (lowercase)
        """
        tax_type_lower = taxonomy_type.lower()
        
        # Check explicit mapping
        if tax_type_lower in self.type_mappings:
            return self.type_mappings[tax_type_lower]
        
        # Fallback: use lowercase taxonomy type as expected NER type
        return {tax_type_lower}
    
    def check_type_match(
        self,
        ner_type: str,
        taxonomy_type: str,
        taxonomy_id: Optional[str] = None
    ) -> bool:
        """
        Check if NER entity type matches taxonomy concept type.
        
        Args:
            ner_type: Entity type from NER (e.g., "gene", "EnergyType")
            taxonomy_type: Type from taxonomy (e.g., "Gene", "Storage")
            taxonomy_id: Optional concept ID (for looking up type if not provided)
            
        Returns:
            True if types match, False otherwise
        """
        if not self.enabled:
            return True
        
        # If taxonomy_type not provided, try to look it up
        if not taxonomy_type and taxonomy_id and taxonomy_id in self.taxonomy_types:
            taxonomy_type = self.taxonomy_types[taxonomy_id]
        
        if not taxonomy_type:
            # No type information available, allow the linking
            return True
        
        ner_type_lower = ner_type.lower()
        expected_types = self.get_expected_ner_types(taxonomy_type)
        
        return ner_type_lower in expected_types
    
    def filter_linking_by_type(
        self,
        entity_type: str,
        linking_result: Optional[List[Dict]],
        entity_text: str = ""
    ) -> Tuple[Optional[List[Dict]], bool]:
        """
        Filter linking result by type match.
        
        Args:
            entity_type: NER entity type
            linking_result: Linking result (list of dicts with 'id', 'name', etc.)
            entity_text: Entity text (for logging)
            
        Returns:
            Tuple of (filtered_linking, was_rejected)
            - If type matches: (linking_result, False)
            - If type mismatch: (None, True)
            - If linking_result is None: (None, False)
        """
        if not self.enabled or not linking_result:
            return linking_result, False
        
        # Get the primary linking (first item)
        primary = linking_result[0]
        linked_id = primary.get('id', '')
        linked_name = primary.get('name', '')
        
        # Try to get taxonomy type from the linking result or lookup
        taxonomy_type = primary.get('type', '')
        
        if not taxonomy_type and linked_id in self.taxonomy_types:
            taxonomy_type = self.taxonomy_types[linked_id]
        
        if not taxonomy_type:
            # No type information, allow the linking
            return linking_result, False
        
        # Check type match
        if self.check_type_match(entity_type, taxonomy_type, linked_id):
            return linking_result, False
        
        # Type mismatch - reject linking
        if self.logger:
            self.logger.debug(
                f"Type mismatch: '{entity_text}' (NER type: {entity_type}) "
                f"linked to {linked_id} (taxonomy type: {taxonomy_type}) - rejected"
            )
        
        return None, True
    
    def get_taxonomy_type(self, taxonomy_id: str) -> Optional[str]:
        """
        Get the type for a taxonomy concept ID.
        
        Args:
            taxonomy_id: Concept identifier
            
        Returns:
            Type string or None if not found
        """
        return self.taxonomy_types.get(taxonomy_id)
    
    def add_taxonomy_types(self, types_dict: Dict[str, str]):
        """
        Add taxonomy types from a dictionary.
        
        Useful for adding types from FTS5 linker results.
        
        Args:
            types_dict: Dict mapping id → type
        """
        self.taxonomy_types.update(types_dict)


class MultiTypeMatcher:
    """
    Manager for multiple TypeMatchers (one per entity type).
    
    Used when different entity types have different taxonomies,
    like in the cancer domain with FTS5 linkers.
    """
    
    def __init__(
        self,
        domain_conf: dict,
        fts5_config: Optional[Dict[str, dict]] = None,
        logger=None
    ):
        """
        Initialize MultiTypeMatcher.
        
        Args:
            domain_conf: Domain configuration from DOMAIN_MODELS
            fts5_config: FTS5 linker configuration (per entity type)
            logger: Optional logger instance
        """
        self.logger = logger
        self.domain_conf = domain_conf
        self.enabled = domain_conf.get("enforce_type_match", True)
        
        # Global type mappings
        self.type_mappings = {
            k.lower(): self._normalize_mapping_value(v)
            for k, v in domain_conf.get("type_mappings", {}).items()
        }
        
        # Per-entity-type matchers (for FTS5)
        self.matchers: Dict[str, TypeMatcher] = {}
        
        if fts5_config:
            for entity_type, config in fts5_config.items():
                taxonomy_path = config.get("taxonomy_path")
                if taxonomy_path:
                    self.matchers[entity_type.lower()] = TypeMatcher(
                        domain_conf=domain_conf,
                        taxonomy_path=taxonomy_path,
                        logger=logger
                    )
        
        if self.logger and self.enabled:
            self.logger.info(f"✅ MultiTypeMatcher initialized for {len(self.matchers)} entity types")
    
    def _normalize_mapping_value(self, value: Union[str, List[str]]) -> Set[str]:
        """Normalize mapping value to set of lowercase strings."""
        if isinstance(value, str):
            return {value.lower()}
        elif isinstance(value, (list, tuple, set)):
            return {str(v).lower() for v in value}
        else:
            return {str(value).lower()}
    
    def check_type_match(
        self,
        ner_type: str,
        taxonomy_type: str,
        entity_type: Optional[str] = None
    ) -> bool:
        """
        Check if NER type matches taxonomy type.
        
        Args:
            ner_type: Entity type from NER
            taxonomy_type: Type from taxonomy
            entity_type: Entity type for selecting matcher (optional)
            
        Returns:
            True if types match
        """
        if not self.enabled:
            return True
        
        # Use entity-specific matcher if available
        if entity_type and entity_type.lower() in self.matchers:
            return self.matchers[entity_type.lower()].check_type_match(
                ner_type, taxonomy_type
            )
        
        # Use global type mappings
        if not taxonomy_type:
            return True
        
        ner_type_lower = ner_type.lower()
        tax_type_lower = taxonomy_type.lower()
        
        expected = self.type_mappings.get(tax_type_lower, {tax_type_lower})
        return ner_type_lower in expected
    
    def filter_linking_by_type(
        self,
        entity_type: str,
        linking_result: Optional[List[Dict]],
        entity_text: str = ""
    ) -> Tuple[Optional[List[Dict]], bool]:
        """
        Filter linking result by type match.
        
        Args:
            entity_type: NER entity type
            linking_result: Linking result
            entity_text: Entity text (for logging)
            
        Returns:
            Tuple of (filtered_linking, was_rejected)
        """
        if not self.enabled or not linking_result:
            return linking_result, False
        
        entity_type_lower = entity_type.lower()
        
        # Use entity-specific matcher if available
        if entity_type_lower in self.matchers:
            return self.matchers[entity_type_lower].filter_linking_by_type(
                entity_type, linking_result, entity_text
            )
        
        # Use global type checking
        primary = linking_result[0]
        taxonomy_type = primary.get('type', '')
        
        if not taxonomy_type:
            return linking_result, False
        
        if self.check_type_match(entity_type, taxonomy_type):
            return linking_result, False
        
        if self.logger:
            self.logger.debug(
                f"Type mismatch: '{entity_text}' (NER: {entity_type}) "
                f"→ taxonomy type: {taxonomy_type} - rejected"
            )
        
        return None, True


# =============================================================================
# Standalone Functions (for simpler use cases)
# =============================================================================

def load_taxonomy_types(
    taxonomy_path: str,
    type_column: str = "type",
    id_column: str = "id"
) -> Dict[str, str]:
    """
    Load concept types from taxonomy TSV file.
    
    Args:
        taxonomy_path: Path to taxonomy TSV
        type_column: Name of type column
        id_column: Name of ID column
        
    Returns:
        Dict mapping concept_id → type
    """
    types = {}
    
    try:
        df = pd.read_csv(taxonomy_path, sep='\t', dtype=str).fillna("")
        
        if id_column not in df.columns or type_column not in df.columns:
            return types
        
        for _, row in df.iterrows():
            concept_id = str(row[id_column]).strip()
            concept_type = str(row[type_column]).strip()
            if concept_id and concept_type:
                types[concept_id] = concept_type
                
    except Exception:
        pass
    
    return types


def check_type_match(
    ner_type: str,
    taxonomy_type: str,
    type_mappings: Optional[Dict[str, Union[str, List[str]]]] = None
) -> bool:
    """
    Simple function to check if NER type matches taxonomy type.
    
    Args:
        ner_type: Entity type from NER
        taxonomy_type: Type from taxonomy
        type_mappings: Optional mapping of taxonomy_type → expected NER type(s)
        
    Returns:
        True if types match
    """
    if not taxonomy_type:
        return True
    
    ner_lower = ner_type.lower()
    tax_lower = taxonomy_type.lower()
    
    if type_mappings:
        mappings_normalized = {
            k.lower(): (
                {v.lower()} if isinstance(v, str) 
                else {x.lower() for x in v}
            )
            for k, v in type_mappings.items()
        }
        
        if tax_lower in mappings_normalized:
            return ner_lower in mappings_normalized[tax_lower]
    
    # Fallback: direct comparison
    return ner_lower == tax_lower


def filter_linking_by_type(
    entity_type: str,
    linking_result: Optional[List[Dict]],
    type_mappings: Optional[Dict[str, Union[str, List[str]]]] = None,
    taxonomy_types: Optional[Dict[str, str]] = None
) -> Tuple[Optional[List[Dict]], bool]:
    """
    Filter linking result by type match.
    
    Args:
        entity_type: NER entity type
        linking_result: Linking result
        type_mappings: Mapping of taxonomy_type → expected NER type(s)
        taxonomy_types: Mapping of concept_id → type
        
    Returns:
        Tuple of (filtered_linking, was_rejected)
    """
    if not linking_result:
        return None, False
    
    primary = linking_result[0]
    linked_id = primary.get('id', '')
    
    # Get taxonomy type
    taxonomy_type = primary.get('type', '')
    if not taxonomy_type and taxonomy_types and linked_id in taxonomy_types:
        taxonomy_type = taxonomy_types[linked_id]
    
    if not taxonomy_type:
        return linking_result, False
    
    if check_type_match(entity_type, taxonomy_type, type_mappings):
        return linking_result, False
    
    return None, True


# =============================================================================
# CLI for Testing
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test type matching")
    parser.add_argument("-e", "--entity-type", required=True, help="NER entity type")
    parser.add_argument("-t", "--taxonomy-type", required=True, help="Taxonomy type")
    parser.add_argument("-d", "--domain", default=None, help="Domain name (for mappings)")
    parser.add_argument("--mappings", default=None, help="JSON file with type mappings")
    
    args = parser.parse_args()
    
    # Load mappings
    type_mappings = None
    
    if args.domain:
        # Try to load from domain_models
        try:
            from configs.domain_models import DOMAIN_MODELS
            domain_conf = DOMAIN_MODELS.get(args.domain, {})
            type_mappings = domain_conf.get("type_mappings", {})
        except ImportError:
            pass
    
    if args.mappings:
        import json
        with open(args.mappings) as f:
            type_mappings = json.load(f)
    
    # Check match
    is_match = check_type_match(
        args.entity_type,
        args.taxonomy_type,
        type_mappings
    )
    
    print(f"NER type:      {args.entity_type}")
    print(f"Taxonomy type: {args.taxonomy_type}")
    print(f"Match:         {'✓ Yes' if is_match else '✗ No'}")
    
    if type_mappings:
        tax_lower = args.taxonomy_type.lower()
        if tax_lower in {k.lower() for k in type_mappings}:
            expected = type_mappings.get(args.taxonomy_type) or type_mappings.get(tax_lower)
            print(f"Expected NER:  {expected}")
