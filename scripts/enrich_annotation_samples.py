#!/usr/bin/env python3
"""
Enrich annotation samples with:
- Score NER: NER confidence score from the JSONL output
- Score EL: Entity linking score from the JSONL output  
- Taxonomy type: Type of the linked concept from the taxonomy
- Type match: Whether entity_type matches taxonomy_type (after normalization)

The `id` column in the samples has format:
    {short_id}#offset_{start}_{end}#{mention_start}-{mention_end}
    Example: 966dc72a#offset_7284_13013#2819-2848

This maps to:
    - section_id: http://scilake-project.eu/res/{short_id}#offset_{start}_{end}
    - entity start: mention_start
    - entity end: mention_end
"""

import json
import pandas as pd
import re
from pathlib import Path
import sys


# =============================================================================
# TYPE MAPPING CONFIGURATION
# =============================================================================
# Maps taxonomy types to expected NER entity types (normalized to lowercase)
# This handles cases where:
#   1. Case differences (e.g., "Gene" vs "gene")
#   2. Semantic mappings (e.g., "Renewables" -> "energytype")

TYPE_MAPPINGS = {
    # Energy domain: taxonomy categorizes energy source, NER categorizes entity type
    "energy": {
        "Storage": "energystorage",
        "Renewables": "energytype",
        "Non-renewable": "energytype", 
        "Total energy": "energytype",
    },
    
    # Cancer domain: simple case normalization
    "cancer": {
        "Gene": "gene",
        "Disease": "disease",
        "CellLine": "cellline",
        "Chemical": "chemical",
        "Species": "species",
        "Variant": "variant",
    },
    
    # Neuro domain: simple case normalization
    "neuro": {
        "UBERONParcellation": "uberonparcellation",
        "technique": "technique",
        "species": "species",
        "preparationType": "preparationtype",
        "biologicalSex": "biologicalsex",
    },
    
    # Maritime domain: already matching
    "maritime": {
        "vesselType": "vesseltype",
    },
    
    # CCAM domain: mappings based on SINFONICA and FAME taxonomies
    # NER types: automation technologies, communication types, entity connection types,
    #            scenario types, sensor types, vehicle types, VRU types
    "ccam": {
        # === SINFONICA descriptive types ===
        "Terms related to automation capabilities of vehicles": "automation technologies",
        "Terms related to vehicles equipment": "sensor types",
        # V2X/V2V = entity connection, C-V2X/DSRC = communication - allow both
        "Terms related to data, communication & connectivity": ["entity connection types", "communication types"],
        "Terms related to CCAM services": "scenario types",
        "Terms related to infrastructure and management": "scenario types",
        "Terms related to social aspects CCAM users": "vru types",
        "Terms related to CCAM legislation": "scenario types",
        "Terms related to CCAM deployment": "scenario types",
        "General terms related to CCAM": "automation technologies",
        
        # === FAME simple category types ===
        "AI": "automation technologies",
        "Vehicle": "vehicle types",
        "Technology": "sensor types",
        "User": "vru types",
        "Data": ["entity connection types", "communication types"],
        "Infrastructure": "scenario types",
        "Operation": "scenario types",
        "Safety": "scenario types",
        "Evaluation": "scenario types",
        "Regulation": "scenario types",
        
        # === FAME multi-category types ===
        # Vehicle combinations → vehicle types
        "Vehicle|Technology": "vehicle types",
        "Vehicle|Operation": "vehicle types",
        "Vehicle|Technology|Infrastructure": "vehicle types",
        "Vehicle|Operation|Technology": "vehicle types",
        "Vehicle|Operation|User|Technology": "vehicle types",
        "Vehicle|Technology|Evaluation|Safety": "vehicle types",
        
        # Vehicle|User combinations → vehicle types (remote operators, etc.)
        "Vehicle|User": "vehicle types",
        "Vehicle|User|Technology": "vehicle types",
        "Vehicle|User|Technology|Regulation": "vehicle types",
        
        # Safety|Operation → scenario types
        "Safety|Operation": "scenario types",
        "Safety|Vehicle": "scenario types",
        
        # Evaluation|Users → vru types (drivers, users)
        "Evaluation|Users": "vru types",
        
        # Data|Evaluation → scenario types (test cases, evaluations)
        "Data|Evaluation": "scenario types",
        "Data|Evaluation|Safety": "scenario types",
        "Data|Evaluation|Safety|Technology": "scenario types",
        
        # Operation combinations → scenario types
        "Operation|Vehicle": "scenario types",
        
        # Infrastructure combinations → scenario types
        "Infrastructure|Vehicle|Technology": "scenario types",
        "Infrastructure|Vehicle|Technology|Operation": "scenario types",
        "Infrastructure|Technology": "scenario types",
        "Infrastructure|Data": ["entity connection types", "communication types"],
        
        # CDA device types (connected automated driving) → vehicle types
        "Vehicle|Technology|Infrastructure": "vehicle types",
        "CDA device": "vehicle types",
    },
}


def get_expected_ner_type(taxonomy_type: str, domain: str = None):
    """
    Get the expected NER entity type(s) for a given taxonomy type.
    
    Args:
        taxonomy_type: The type from the taxonomy file
        domain: Optional domain name for domain-specific mappings
        
    Returns:
        Expected NER type (lowercase), list of valid types, or None if no mapping
    """
    if pd.isna(taxonomy_type) or not taxonomy_type:
        return None
    
    taxonomy_type = str(taxonomy_type).strip()
    
    # Try domain-specific mapping first
    if domain and domain in TYPE_MAPPINGS:
        if taxonomy_type in TYPE_MAPPINGS[domain]:
            return TYPE_MAPPINGS[domain][taxonomy_type]
    
    # Try all domain mappings
    for domain_mappings in TYPE_MAPPINGS.values():
        if taxonomy_type in domain_mappings:
            return domain_mappings[taxonomy_type]
    
    # CCAM fallback: handle multi-category types not explicitly mapped
    if domain == "ccam" and "|" in taxonomy_type:
        # Get the first category
        first_category = taxonomy_type.split("|")[0].strip()
        
        # Map based on first category
        ccam_category_map = {
            "Vehicle": "vehicle types",
            "User": "vru types",
            "Data": ["entity connection types", "communication types"],  # Could be either
            "Technology": "sensor types",
            "AI": "automation technologies",
            "Operation": "scenario types",
            "Infrastructure": "scenario types",
            "Safety": "scenario types",
            "Evaluation": "scenario types",
            "Regulation": "scenario types",
        }
        
        if first_category in ccam_category_map:
            return ccam_category_map[first_category]
    
    # CCAM fallback for SINFONICA descriptive types not explicitly mapped
    if domain == "ccam":
        taxonomy_lower = taxonomy_type.lower()
        if "automation" in taxonomy_lower:
            return "automation technologies"
        elif "equipment" in taxonomy_lower or "sensor" in taxonomy_lower:
            return "sensor types"
        elif "communication" in taxonomy_lower or "connectivity" in taxonomy_lower:
            # Could be communication types OR entity connection types
            return ["entity connection types", "communication types"]
        elif "user" in taxonomy_lower or "social" in taxonomy_lower:
            return "vru types"
        elif "vehicle" in taxonomy_lower:
            return "vehicle types"
    
    # Default: lowercase the taxonomy type
    return taxonomy_type.lower()


def check_type_match(entity_type: str, taxonomy_type: str, domain: str = None) -> str:
    """
    Check if entity_type matches taxonomy_type (after normalization).
    
    Args:
        entity_type: The entity type from NER
        taxonomy_type: The type from taxonomy
        domain: Domain name for domain-specific mappings
    
    Returns:
        'yes' - types match (or entity_type is one of the valid options)
        'no' - types don't match
        'na' - cannot determine (missing data or no mapping)
    """
    if pd.isna(entity_type) or not entity_type:
        return "na"
    if pd.isna(taxonomy_type) or not taxonomy_type:
        return "na"
    
    entity_type_norm = str(entity_type).lower().strip()
    expected_type = get_expected_ner_type(taxonomy_type, domain)
    
    if expected_type is None:
        return "na"
    
    # Handle case where multiple types are valid
    if isinstance(expected_type, list):
        return "yes" if entity_type_norm in expected_type else "no"
    
    return "yes" if entity_type_norm == expected_type else "no"


def parse_sample_id(sample_id: str) -> tuple:
    """
    Parse the sample ID to extract section_id and entity position.
    
    Handles two formats:
    1. Short ID: "966dc72a#offset_7284_13013#2819-2848"
    2. Full URI: "http://scilake-project.eu/res/e3a8013e#offset_21858_22409#100-110"
    
    Returns: (section_id, start, end)
    """
    # Check if it's already a full URI
    uri_prefix = "http://scilake-project.eu/res/"
    
    if sample_id.startswith(uri_prefix):
        # Full URI format - extract the section_id directly
        # Split only on the last # to get position
        last_hash_idx = sample_id.rfind('#')
        if last_hash_idx > 0:
            section_id = sample_id[:last_hash_idx]
            position_part = sample_id[last_hash_idx + 1:]
            
            start_end = position_part.split('-')
            if len(start_end) == 2:
                try:
                    start = int(start_end[0])
                    end = int(start_end[1])
                    return section_id, start, end
                except ValueError:
                    pass
    else:
        # Short ID format: "966dc72a#offset_7284_13013#2819-2848"
        parts = sample_id.split('#')
        if len(parts) >= 3:
            short_id = parts[0]
            offset_part = parts[1]  # e.g., "offset_7284_13013"
            position_part = parts[2]  # e.g., "2819-2848"
            
            section_id = f"{uri_prefix}{short_id}#{offset_part}"
            
            start_end = position_part.split('-')
            if len(start_end) == 2:
                try:
                    start = int(start_end[0])
                    end = int(start_end[1])
                    return section_id, start, end
                except ValueError:
                    pass
    
    return None, None, None


def load_taxonomy(taxonomy_path: str) -> dict:
    """
    Load taxonomy and create mapping from concept ID to type.
    Returns dict: {concept_id: type}
    """
    df = pd.read_csv(taxonomy_path, sep='\t')
    return dict(zip(df['id'].astype(str), df['type'].astype(str)))


def load_jsonl_entities(jsonl_path: str) -> dict:
    """
    Load entities from JSONL file and organize by section_id.
    Returns: {section_id: [list of entities]}
    """
    entities_by_section = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                section_id = record.get('section_id', '')
                entities = record.get('entities', [])
                entities_by_section[section_id] = entities
    
    return entities_by_section


def find_matching_entity(entities: list, mention: str, start: int, end: int, model_name: str) -> dict:
    """
    Find the matching entity in the list based on text, position, and model.
    Returns the matching entity dict or None.
    """
    for entity in entities:
        # Match by position and text
        if entity.get('start') == start and entity.get('end') == end:
            if entity.get('text', '').lower() == mention.lower():
                # If model name is specified, try to match it
                entity_model = entity.get('model', '')
                if model_name:
                    # Check if model names match (could be partial match)
                    if model_name in entity_model or entity_model in model_name:
                        return entity
                    # If exact model not found, continue looking
                else:
                    return entity
    
    # Fallback: try to find by text and approximate position (within ±1)
    for entity in entities:
        entity_start = entity.get('start', -999)
        entity_end = entity.get('end', -999)
        if abs(entity_start - start) <= 1 and abs(entity_end - end) <= 1:
            if entity.get('text', '').lower() == mention.lower():
                return entity
    
    return None


def get_el_score(entity: dict, concept_id: str) -> float:
    """
    Extract the EL score for a specific concept_id from the linking results.
    
    Note: Gazetteer (exact match) results don't have EL scores - they're implicit 1.0
    Only semantic/neural linkers (GLiNER, etc.) have EL scores.
    """
    linking = entity.get('linking', [])
    if not linking:
        return None
    
    for link in linking:
        if link.get('id') == concept_id:
            return link.get('score')
    
    # If no exact match, return the score from the first link (if any)
    if linking and 'score' in linking[0]:
        return linking[0].get('score')
    
    # For Gazetteer matches, return None (they don't have explicit scores)
    return None


def enrich_samples(samples_path: str, taxonomy_path: str, el_dir: str, output_path: str, domain: str = None):
    """
    Main function to enrich annotation samples.
    
    Args:
        samples_path: Path to annotation samples TSV
        taxonomy_path: Path to taxonomy TSV
        el_dir: Directory containing EL output JSONL files
        output_path: Path for enriched output TSV
        domain: Domain name for type mapping (neuro, cancer, energy, maritime, ccam)
    """
    print(f"Loading annotation samples from: {samples_path}")
    df = pd.read_csv(samples_path, sep='\t')
    print(f"Loaded {len(df)} samples")
    
    if domain:
        print(f"Using domain-specific type mappings for: {domain}")
    
    print(f"Loading taxonomy from: {taxonomy_path}")
    taxonomy_types = load_taxonomy(taxonomy_path)
    print(f"Loaded {len(taxonomy_types)} taxonomy entries")
    
    # Initialize new columns
    df['score_ner'] = None
    df['score_el'] = None
    df['taxonomy_type'] = None
    df['type_match'] = None
    
    # Cache for loaded JSONL files
    jsonl_cache = {}
    
    # Process each sample
    for idx, row in df.iterrows():
        sample_id = row['id']
        mention = row['mention']
        model_name = row.get('linking_model', '')
        concept_id = row.get('linked_to_concept_id', '')
        source_file = row.get('source_file', '')
        
        # Parse sample ID
        section_id, start, end = parse_sample_id(sample_id)
        
        if section_id is None:
            print(f"Warning: Could not parse sample ID: {sample_id}")
            continue
        
        # Construct JSONL path
        # The source_file already contains :: separators matching the actual filename
        jsonl_filename = source_file + '.jsonl'
        jsonl_path = Path(el_dir) / jsonl_filename
        
        # Load JSONL if not cached
        if jsonl_path not in jsonl_cache:
            if jsonl_path.exists():
                print(f"Loading: {jsonl_path}")
                jsonl_cache[jsonl_path] = load_jsonl_entities(str(jsonl_path))
            else:
                print(f"Warning: JSONL file not found: {jsonl_path}")
                jsonl_cache[jsonl_path] = {}
        
        entities_by_section = jsonl_cache[jsonl_path]
        
        # Find the section
        if section_id not in entities_by_section:
            print(f"Warning: Section not found: {section_id}")
            continue
        
        entities = entities_by_section[section_id]
        
        # Find matching entity
        entity = find_matching_entity(entities, mention, start, end, model_name)
        
        if entity:
            # Get NER score
            df.at[idx, 'score_ner'] = entity.get('score')
            
            # Get EL score
            if concept_id and concept_id != '[UNLINKED]':
                el_score = get_el_score(entity, concept_id)
                df.at[idx, 'score_el'] = el_score
            
            # Get taxonomy type
            if concept_id and concept_id != '[UNLINKED]':
                taxonomy_type = taxonomy_types.get(concept_id, '')
                df.at[idx, 'taxonomy_type'] = taxonomy_type
                
                # Check type match
                entity_type = row.get('entity_type', '')
                df.at[idx, 'type_match'] = check_type_match(entity_type, taxonomy_type, domain)
        else:
            print(f"Warning: Entity not found - {mention} at {start}-{end} in {section_id}")
    
    # Save enriched samples
    df.to_csv(output_path, sep='\t', index=False)
    print(f"\nEnriched samples saved to: {output_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total samples: {len(df)}")
    print(f"Samples with NER score: {df['score_ner'].notna().sum()}")
    print(f"Samples with EL score: {df['score_el'].notna().sum()}")
    print(f"Samples with taxonomy type: {df['taxonomy_type'].notna().sum()}")
    
    # Type match summary
    type_match_counts = df['type_match'].value_counts()
    print(f"\nType match results:")
    print(f"  - Match (yes): {type_match_counts.get('yes', 0)}")
    print(f"  - No match (no): {type_match_counts.get('no', 0)}")
    print(f"  - N/A: {type_match_counts.get('na', 0)}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enrich annotation samples with NER scores, EL scores, taxonomy types, and type matching.',
        epilog='''
Example usage:
  python enrich_annotation_samples.py \\
    --samples /root/scilake-enrichments/outputs/neuro-all-ft/neuro_el_annotation_samples.tsv \\
    --taxonomy /root/scilake-enrichments/taxonomies/neuro/Neuroscience_Combined.tsv \\
    --el-dir /root/scilake-enrichments/outputs/neuro-all-ft/el \\
    --output /root/scilake-enrichments/outputs/neuro-all-ft/neuro_el_annotation_samples_enriched.tsv \\
    --domain neuro

Notes:
  - Score NER: The confidence score from the NER model (1.0 for Gazetteer exact matches)
  - Score EL: The entity linking score (only present for semantic/neural matches, not Gazetteer)
  - Taxonomy type: The 'type' column from the taxonomy file for the linked concept
  - Type match: Whether entity_type matches taxonomy_type after normalization (yes/no/na)

Supported domains: neuro, cancer, energy, maritime, ccam
        '''
    )
    
    parser.add_argument('--samples', required=True,
                        help='Path to the annotation samples TSV file')
    parser.add_argument('--taxonomy', required=True,
                        help='Path to the taxonomy TSV file')
    parser.add_argument('--el-dir', required=True,
                        help='Directory containing the EL output JSONL files')
    parser.add_argument('--output', required=True,
                        help='Path for the enriched output TSV file')
    parser.add_argument('--domain', required=False, default=None,
                        choices=['neuro', 'cancer', 'energy', 'maritime', 'ccam'],
                        help='Domain name for type mapping (enables domain-specific type normalization)')
    
    args = parser.parse_args()
    
    enrich_samples(args.samples, args.taxonomy, args.el_dir, args.output, args.domain)
