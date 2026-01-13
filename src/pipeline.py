"""
SciLake NER & Entity Linking Pipeline

Updates included:
- Title/abstract JSON input format support (--input_format title_abstract)
- Domain-level blocked_mentions (works with all linking strategies)
- Domain-level min_mention_length filtering
"""

import argparse
import os
import pandas as pd
import traceback
import resource
from typing import Dict, Set, Optional, Tuple, List

from tqdm import tqdm
import torch

from src.utils.logger import setup_logger
from src.utils.io_utils import load_json, save_json, append_jsonl
from src.nif_reader import parse_nif_file, apply_acronym_expansion
from src.ner_runner import predict_sections_multimodel
from src.gazetteer_linker import GazetteerLinker
from src.title_abstract_reader import parse_title_abstract_file, parse_title_abstract_directory
from src.legal_text_reader import parse_legal_text_file, parse_legal_text_directory
from configs.domain_models import DOMAIN_MODELS


# ============================================================
# Utility Functions
# ============================================================

def log_memory_usage(logger, label=""):
    """Log current memory usage (Linux only)"""
    try:
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"📊 Memory {label}: {mem_mb:.2f} MB")
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**2)
                reserved = torch.cuda.memory_reserved(i) / (1024**2)
                logger.info(f"   GPU {i}: {allocated:.2f} MB allocated, {reserved:.2f} MB reserved")
    except Exception as e:
        logger.debug(f"Could not log memory: {e}")


def log_cache_stats(cache: dict, logger):
    """Log cache statistics"""
    if not cache:
        return
    
    total = len(cache)
    rejected = sum(1 for v in cache.values() if v.get('debug', {}).get('rejected', False))
    linked = total - rejected
    
    logger.info(f"📊 Cache stats: {total} total, {linked} linked ({100*linked/total:.1f}%), {rejected} rejected ({100*rejected/total:.1f}%)")


def merge_gazetteer_and_ner(df_gaz, df_ner, logger):
    """Merge gazetteer and NER entities, removing overlaps."""
    merged = []
    
    for section_id in set(df_gaz['section_id'].tolist() + df_ner['section_id'].tolist()):
        gaz_ents = df_gaz[df_gaz['section_id'] == section_id]['entities'].tolist()
        ner_ents = df_ner[df_ner['section_id'] == section_id]['entities'].tolist()
        
        gaz_ents = gaz_ents[0] if gaz_ents else []
        ner_ents = ner_ents[0] if ner_ents else []
        
        combined = gaz_ents.copy()
        for ner_ent in ner_ents:
            if not _overlaps_with_gazetteer(ner_ent, gaz_ents):
                combined.append(ner_ent)
        
        merged.append({'section_id': section_id, 'entities': combined})
    
    return pd.DataFrame(merged)


def _overlaps_with_gazetteer(ner_ent, gaz_ents):
    """Check if NER entity overlaps with any gazetteer entity."""
    ner_start, ner_end = ner_ent['start'], ner_ent['end']
    for gaz_ent in gaz_ents:
        gaz_start, gaz_end = gaz_ent['start'], gaz_ent['end']
        if not (ner_end <= gaz_start or ner_start >= gaz_end):
            return True
    return False


def get_expanded_csv_path(ttl_path: str, expanded_dir: str) -> str:
    """Get the path to the expanded sections CSV for a given TTL file."""
    base_name = os.path.basename(ttl_path).replace(".ttl", "_sections.csv")
    return os.path.join(expanded_dir, base_name)


def load_expanded_sections(csv_path: str, logger=None) -> pd.DataFrame:
    """Load pre-expanded sections from CSV file."""
    try:
        df = pd.read_csv(
            csv_path,
            engine='python',
            on_bad_lines='warn',
            encoding='utf-8',
            dtype=str
        )
        if logger:
            logger.debug(f"📂 Loaded {len(df)} pre-expanded sections from {os.path.basename(csv_path)}")
        return df
    except Exception as e:
        if logger:
            logger.error(f"❌ Failed to load expanded sections from {csv_path}: {e}")
        return None


# ============================================================
# Entity Filtering Functions (NEW)
# ============================================================

def load_entity_filters(domain_conf: dict, logger=None) -> dict:
    """
    Load entity filtering configuration from domain config.
    
    Loads:
    - blocked_mentions: Terms to skip (per entity type or global)
    - min_mention_length: Minimum character length for mentions
    
    Configuration examples in domain_models.py:
    
        # Global minimum length (applies to all entity types)
        "min_mention_length": 2,
        
        # Per-entity-type minimum length
        "min_mention_length": {
            "gene": 2,
            "disease": 3,
            "_default": 2,
        },
        
        # Blocked mentions (flat set - applies to all)
        "blocked_mentions": {"patient", "data", "system"},
        
        # Blocked mentions (per entity type)
        "blocked_mentions": {
            "species": {"patient", "patients"},
            "disease": {"pain", "fever"},
        },
    
    Returns:
        Dict with keys:
        - "blocked_mentions": Dict[str, Set[str]]
        - "min_mention_length": Dict[str, int]
    """
    filters = {
        "blocked_mentions": {"_all": set()},
        "min_mention_length": {"_default": 1},
    }
    
    # === Load blocked mentions ===
    domain_blocked = domain_conf.get("blocked_mentions", {})
    
    if isinstance(domain_blocked, set):
        filters["blocked_mentions"]["_all"] = {str(m).lower() for m in domain_blocked}
        if logger:
            logger.info(f"📋 Blocked mentions: {len(filters['blocked_mentions']['_all'])} terms (all types)")
    
    elif isinstance(domain_blocked, dict):
        for entity_type, mentions in domain_blocked.items():
            entity_type_lower = entity_type.lower()
            if entity_type_lower not in filters["blocked_mentions"]:
                filters["blocked_mentions"][entity_type_lower] = set()
            if isinstance(mentions, (set, list, tuple)):
                filters["blocked_mentions"][entity_type_lower].update(str(m).lower() for m in mentions)
        
        if logger:
            for et, mentions in filters["blocked_mentions"].items():
                if et != "_all" and mentions:
                    logger.info(f"📋 Blocked mentions for '{et}': {len(mentions)} terms")
    
    # Also load from fts5_linkers (backward compatibility)
    fts5_config = domain_conf.get("fts5_linkers", {})
    for entity_type, config in fts5_config.items():
        entity_type_lower = entity_type.lower()
        linker_blocked = config.get("blocked_mentions", set())
        if linker_blocked:
            if entity_type_lower not in filters["blocked_mentions"]:
                filters["blocked_mentions"][entity_type_lower] = set()
            filters["blocked_mentions"][entity_type_lower].update(str(m).lower() for m in linker_blocked)
    
    # === Load min_mention_length ===
    min_length = domain_conf.get("min_mention_length", 1)
    
    if isinstance(min_length, int):
        filters["min_mention_length"]["_default"] = min_length
        if logger and min_length > 1:
            logger.info(f"📏 Min mention length: {min_length} chars (all types)")
    
    elif isinstance(min_length, dict):
        for entity_type, length in min_length.items():
            entity_type_lower = entity_type.lower()
            if isinstance(length, int) and length > 0:
                filters["min_mention_length"][entity_type_lower] = length
        
        if logger:
            non_default = {k: v for k, v in filters["min_mention_length"].items() if k != "_default"}
            if non_default:
                logger.info(f"📏 Min mention length: {non_default}")
    
    return filters


def should_skip_entity(
    entity_text: str,
    entity_type: str,
    filters: dict
) -> Tuple[bool, Optional[str]]:
    """
    Check if an entity mention should be skipped.
    
    Returns:
        Tuple of (should_skip: bool, reason: str or None)
        reason is one of: "blocked", "too_short", None
    """
    text_clean = entity_text.strip()
    text_lower = text_clean.lower()
    type_lower = entity_type.lower()
    
    # Check minimum length
    min_lengths = filters.get("min_mention_length", {})
    min_len = min_lengths.get(type_lower, min_lengths.get("_default", 1))
    
    if len(text_clean) < min_len:
        return True, "too_short"
    
    # Check blocked mentions
    blocked = filters.get("blocked_mentions", {})
    
    if text_lower in blocked.get("_all", set()):
        return True, "blocked"
    
    if text_lower in blocked.get(type_lower, set()):
        return True, "blocked"
    
    return False, None


def filter_entities(
    entities: list,
    filters: dict,
    logger=None,
    log_skipped: bool = False
) -> Tuple[list, dict]:
    """
    Filter a list of entities based on configured filters.
    
    Returns:
        Tuple of (filtered_entities, stats)
        stats = {"total": N, "kept": N, "blocked": N, "too_short": N}
    """
    filtered = []
    stats = {"total": len(entities), "kept": 0, "blocked": 0, "too_short": 0}
    
    for entity in entities:
        entity_text = entity.get('text', '')
        entity_type = entity.get('entity', '')
        
        should_skip, reason = should_skip_entity(entity_text, entity_type, filters)
        
        if should_skip:
            stats[reason] = stats.get(reason, 0) + 1
            if log_skipped and logger:
                logger.debug(f"Skipping ({reason}): '{entity_text}' ({entity_type})")
        else:
            filtered.append(entity)
            stats["kept"] += 1
    
    return filtered, stats


# ============================================================
# NER Functions
# ============================================================

def run_ner_title_abstract(
    domain: str,
    input_path: str,
    output_dir: str,
    resume: bool = False,
    batch_size: int = 1000,
    debug: bool = False,
    gazetteer_only: bool = False
):
    """
    Run NER enrichment on title/abstract JSON files.
    
    Features:
    - Combines title and abstract into single section per record (faster processing)
    - Saves results incrementally after each batch (safe for long-running jobs)
    - Supports resume from checkpoint
    
    Args:
        domain: Domain name (ccam, energy, etc.)
        input_path: Path to JSON file or directory containing JSON files
        output_dir: Output directory for NER results
        resume: Resume from checkpoint
        batch_size: Number of sections per batch
        debug: Enable debug logging
        gazetteer_only: Only run gazetteer (skip neural NER)
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_ner_titleabstract", debug=debug)

    checkpoint_file = os.path.join(checkpoint_dir, "processed_sections.json")

    logger.info(f"✅ Starting NER (title/abstract mode) for domain={domain}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output dir: {output_dir}")

    # Determine output file paths
    if os.path.isfile(input_path):
        source_file = os.path.basename(input_path)
    else:
        source_file = f"{domain}_titleabstract.json"
       
    if source_file.endswith('.jsonl'):
        out_filename = source_file.replace('.jsonl', '_ner.jsonl')
    elif source_file.endswith('.json'):
        out_filename = source_file.replace('.json', '_ner.jsonl')
    else:
        out_filename = f"{source_file}_ner.jsonl"
    
    out_path = os.path.join(output_dir, out_filename)
    
    # Sections file for EL step
    sections_dir = os.path.join(os.path.dirname(output_dir), "sections")
    os.makedirs(sections_dir, exist_ok=True)
    
    if source_file.endswith('.jsonl'):
        sections_filename = source_file.replace('.jsonl', '_ner_sections.csv')
    elif source_file.endswith('.json'):
        sections_filename = source_file.replace('.json', '_ner_sections.csv')
    else:
        sections_filename = f"{source_file}_ner_sections.csv"
    
    sections_path = os.path.join(sections_dir, sections_filename)

    # Load checkpoint
    if resume:
        processed = load_json(checkpoint_file, default={})
        logger.info(f"Resuming from checkpoint: {len(processed)} sections already done")
    else:
        processed = {}
        # Clear output file if not resuming
        if os.path.exists(out_path):
            os.remove(out_path)
        logger.info(f"Processing all sections (resume=False)")

    # Parse input (file or directory) - using combined mode (title + abstract in single section)
    if os.path.isfile(input_path):
        df_all = parse_title_abstract_file(input_path, combine_sections=True, logger=logger)
    else:
        df_all = parse_title_abstract_directory(input_path, domain=domain, combine_sections=True, logger=logger)

    if df_all.empty:
        logger.warning("⚠️ No sections found in input")
        return

    logger.info(f"📊 Loaded {len(df_all)} sections total (combined title+abstract mode)")
    
    # Save sections file for EL step (do this early, before filtering)
    # Only save if not resuming or file doesn't exist
    if not os.path.exists(sections_path):
        df_all.to_csv(sections_path, index=False, escapechar='\\')
        logger.info(f"💾 Saved {len(df_all)} sections to {sections_path}")

    # Filter out already processed sections
    if processed:
        df_pending = df_all[~df_all['section_id'].isin(processed.keys())]
        logger.info(f"📊 {len(df_pending)} sections remaining after filtering processed")
    else:
        df_pending = df_all

    if df_pending.empty:
        logger.info("✅ All sections already processed")
        return

    # Initialize gazetteer if enabled
    gazetteer = None
    domain_conf = DOMAIN_MODELS.get(domain)
    if domain_conf and domain_conf.get('gazetteer', {}).get('enabled'):
        gaz_conf = domain_conf['gazetteer']
        gazetteer = GazetteerLinker(
            taxonomy_path=gaz_conf['taxonomy_path'],
            taxonomy_source=gaz_conf.get('taxonomy_source'),
            model_name=gaz_conf.get('model_name'),
            default_type=gaz_conf.get('default_type'),
            domain=domain,
            min_term_length=gaz_conf.get('min_term_length', 2),
            blocked_terms=gaz_conf.get('blocked_terms'),
            logger=logger
        )
        logger.info(f"✅ Gazetteer loaded: {gazetteer.model_name} from {gaz_conf['taxonomy_path']}")

    # Process in batches
    total_sections = len(df_pending)
    total_entities_all = 0
    total_sections_processed = len(processed)

    for start in range(0, total_sections, batch_size):
        batch_df = df_pending.iloc[start:start + batch_size].copy()
        batch_num = start // batch_size + 1
        total_batches = (total_sections + batch_size - 1) // batch_size
        logger.info(f"\n🧩 Processing batch {batch_num}/{total_batches} ({len(batch_df)} sections)...")

        # Run gazetteer
        df_gazetteer = pd.DataFrame()
        if gazetteer:
            logger.info("🔍 Running gazetteer-based entity extraction...")
            gazetteer_entities = []
            for _, row in batch_df.iterrows():
                gaz_ents = gazetteer.extract_entities(
                    text=row['section_content_expanded'],
                    section_id=row['section_id'],
                    domain=domain
                )
                gazetteer_entities.append({
                    'section_id': row['section_id'],
                    'entities': gaz_ents
                })
            df_gazetteer = pd.DataFrame(gazetteer_entities)
            total_gaz = sum(len(e['entities']) for e in gazetteer_entities)
            logger.info(f"✅ Gazetteer found {total_gaz} entities")

        # Run NER models (unless gazetteer_only)
        df_ner = pd.DataFrame()
        if not gazetteer_only:
            logger.info("🤖 Running NER models...")
            df_ner = predict_sections_multimodel(
                batch_df,
                domain=domain,
                logger=logger
            )
            if not df_ner.empty:
                total_ner = sum(len(e) for e in df_ner['entities'].tolist())
                logger.info(f"✅ NER found {total_ner} entities")

        # Merge gazetteer + NER
        if not df_gazetteer.empty and not df_ner.empty:
            df_merged = merge_gazetteer_and_ner(df_gazetteer, df_ner, logger)
        elif not df_gazetteer.empty:
            df_merged = df_gazetteer
        elif not df_ner.empty:
            df_merged = df_ner
        else:
            logger.warning("⚠️ No entities found in this batch")
            for section_id in batch_df['section_id'].tolist():
                processed[section_id] = {"status": "no_entities"}
            save_json(processed, checkpoint_file)
            continue

        # Save results incrementally (append to JSONL)
        batch_results = []
        for _, row in df_merged.iterrows():
            batch_results.append(row.to_dict())
            processed[row['section_id']] = {"status": "done", "entities": len(row['entities'])}
        
        # Append batch results to output file
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            # Append mode: write header only if file doesn't exist
            write_header = not os.path.exists(out_path)
            with open(out_path, 'a', encoding='utf-8') as f:
                for record in batch_results:
                    f.write(pd.Series(record).to_json() + '\n')
            
            batch_entities = sum(len(r['entities']) for r in batch_results)
            total_entities_all += batch_entities
            total_sections_processed += len(batch_results)
            logger.info(f"💾 Appended {len(batch_results)} sections ({batch_entities} entities) to {out_filename}")

        # Save checkpoint
        save_json(processed, checkpoint_file)
        logger.info(f"💾 Checkpoint saved: {len(processed)} sections processed total")

    logger.info(f"\n🎉 Title/abstract NER processing complete.")
    logger.info(f"📊 Total: {total_sections_processed} sections, {total_entities_all} entities")
    logger.info(f"📄 Output: {out_path}")
    logger.info(f"📄 Sections: {sections_path}")


def run_ner_legal_text(
    domain: str,
    input_path: str,
    output_dir: str,
    resume: bool = False,
    batch_size: int = 1000,
    debug: bool = False,
    gazetteer_only: bool = False
):
    """
    Run NER enrichment on legal text JSON files.
    
    Features:
    - Saves results incrementally after each batch (safe for long-running jobs)
    - Supports resume from checkpoint
    - Handles long legal texts (NER chunking is done internally)
    
    Args:
        domain: Domain name (energy, etc.)
        input_path: Path to JSON file or directory containing JSON files
        output_dir: Output directory for NER results
        resume: Resume from checkpoint
        batch_size: Number of sections per batch
        debug: Enable debug logging
        gazetteer_only: Only run gazetteer (skip neural NER)
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_ner_legal_text", debug=debug)

    checkpoint_file = os.path.join(checkpoint_dir, "processed_sections.json")

    logger.info(f"✅ Starting NER (legal text mode) for domain={domain}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output dir: {output_dir}")

    # Determine output file paths
    if os.path.isfile(input_path):
        source_file = os.path.basename(input_path)
    else:
        source_file = f"{domain}_legal_text.json"
    
    # Handle .jsonl extension
    if source_file.endswith('.jsonl'):
        out_filename = source_file.replace('.jsonl', '_ner.jsonl')
    elif source_file.endswith('.json'):
        out_filename = source_file.replace('.json', '_ner.jsonl')
    else:
        out_filename = f"{source_file}_ner.jsonl"
    
    out_path = os.path.join(output_dir, out_filename)
    
    # Sections file for EL step
    sections_dir = os.path.join(os.path.dirname(output_dir), "sections")
    os.makedirs(sections_dir, exist_ok=True)
    
    if source_file.endswith('.jsonl'):
        sections_filename = source_file.replace('.jsonl', '_ner_sections.csv')
    elif source_file.endswith('.json'):
        sections_filename = source_file.replace('.json', '_ner_sections.csv')
    else:
        sections_filename = f"{source_file}_ner_sections.csv"
    
    sections_path = os.path.join(sections_dir, sections_filename)

    # Load checkpoint
    if resume:
        processed = load_json(checkpoint_file, default={})
        logger.info(f"Resuming from checkpoint: {len(processed)} sections already done")
    else:
        processed = {}
        # Clear output file if not resuming
        if os.path.exists(out_path):
            os.remove(out_path)
        logger.info(f"Processing all sections (resume=False)")

    # Parse input (file or directory)
    if os.path.isfile(input_path):
        df_all = parse_legal_text_file(input_path, include_title=True, logger=logger)
    else:
        df_all = parse_legal_text_directory(input_path, include_title=True, logger=logger)

    if df_all.empty:
        logger.warning("⚠️ No sections found in input")
        return

    logger.info(f"📊 Loaded {len(df_all)} sections total")
    
    # Log text length statistics
    lengths = df_all['section_content_expanded'].str.len()
    logger.info(f"   Text lengths - Avg: {lengths.mean():.0f}, Max: {lengths.max()}, Over 1M: {(lengths > 1000000).sum()}")
    
    # Save sections file for EL step (do this early, before filtering)
    if not os.path.exists(sections_path):
        df_all.to_csv(sections_path, index=False, escapechar='\\')
        logger.info(f"💾 Saved {len(df_all)} sections to {sections_path}")

    # Filter out already processed sections
    if processed:
        df_pending = df_all[~df_all['section_id'].isin(processed.keys())]
        logger.info(f"📊 {len(df_pending)} sections remaining after filtering processed")
    else:
        df_pending = df_all

    if df_pending.empty:
        logger.info("✅ All sections already processed")
        return

    # Initialize gazetteer if enabled
    gazetteer = None
    domain_conf = DOMAIN_MODELS.get(domain)
    if domain_conf and domain_conf.get('gazetteer', {}).get('enabled'):
        gaz_conf = domain_conf['gazetteer']
        gazetteer = GazetteerLinker(
            taxonomy_path=gaz_conf['taxonomy_path'],
            taxonomy_source=gaz_conf.get('taxonomy_source'),
            model_name=gaz_conf.get('model_name'),
            default_type=gaz_conf.get('default_type'),
            domain=domain,
            min_term_length=gaz_conf.get('min_term_length', 2),
            blocked_terms=gaz_conf.get('blocked_terms'),
            logger=logger
        )
        logger.info(f"✅ Gazetteer loaded: {gazetteer.model_name} from {gaz_conf['taxonomy_path']}")

    # Process in batches
    total_sections = len(df_pending)
    total_entities_all = 0
    total_sections_processed = len(processed)

    for start in range(0, total_sections, batch_size):
        batch_df = df_pending.iloc[start:start + batch_size].copy()
        batch_num = start // batch_size + 1
        total_batches = (total_sections + batch_size - 1) // batch_size
        logger.info(f"\n🧩 Processing batch {batch_num}/{total_batches} ({len(batch_df)} sections)...")

        # Run gazetteer
        df_gazetteer = pd.DataFrame()
        if gazetteer:
            logger.info("🔍 Running gazetteer-based entity extraction...")
            gazetteer_entities = []
            for _, row in batch_df.iterrows():
                gaz_ents = gazetteer.extract_entities(
                    text=row['section_content_expanded'],
                    section_id=row['section_id'],
                    domain=domain
                )
                gazetteer_entities.append({
                    'section_id': row['section_id'],
                    'entities': gaz_ents
                })
            df_gazetteer = pd.DataFrame(gazetteer_entities)
            total_gaz = sum(len(e['entities']) for e in gazetteer_entities)
            logger.info(f"✅ Gazetteer found {total_gaz} entities")

        # Run NER models (unless gazetteer_only)
        df_ner = pd.DataFrame()
        if not gazetteer_only:
            logger.info("🤖 Running NER models...")
            df_ner = predict_sections_multimodel(
                batch_df,
                domain=domain,
                logger=logger
            )
            if not df_ner.empty:
                total_ner = sum(len(e) for e in df_ner['entities'].tolist())
                logger.info(f"✅ NER found {total_ner} entities")

        # Merge gazetteer + NER
        if not df_gazetteer.empty and not df_ner.empty:
            df_merged = merge_gazetteer_and_ner(df_gazetteer, df_ner, logger)
        elif not df_gazetteer.empty:
            df_merged = df_gazetteer
        elif not df_ner.empty:
            df_merged = df_ner
        else:
            logger.warning("⚠️ No entities found in this batch")
            for section_id in batch_df['section_id'].tolist():
                processed[section_id] = {"status": "no_entities"}
            save_json(processed, checkpoint_file)
            continue

        # Save results incrementally (append to JSONL)
        batch_results = []
        for _, row in df_merged.iterrows():
            batch_results.append(row.to_dict())
            processed[row['section_id']] = {"status": "done", "entities": len(row['entities'])}
        
        # Append batch results to output file
        if batch_results:
            df_batch = pd.DataFrame(batch_results)
            with open(out_path, 'a', encoding='utf-8') as f:
                for record in batch_results:
                    f.write(pd.Series(record).to_json() + '\n')
            
            batch_entities = sum(len(r['entities']) for r in batch_results)
            total_entities_all += batch_entities
            total_sections_processed += len(batch_results)
            logger.info(f"💾 Appended {len(batch_results)} sections ({batch_entities} entities) to {out_filename}")

        # Save checkpoint
        save_json(processed, checkpoint_file)
        logger.info(f"💾 Checkpoint saved: {len(processed)} sections processed total")

    logger.info(f"\n🎉 Legal text NER processing complete.")
    logger.info(f"📊 Total: {total_sections_processed} sections, {total_entities_all} entities")
    logger.info(f"📄 Output: {out_path}")
    logger.info(f"📄 Sections: {sections_path}")


def run_ner(domain, input_dir, output_dir, resume=False, file_batch_size=100, debug=False, gazetter_only=False):
    """
    Run NER enrichment by batching multiple papers together.
    For each batch:
     - Parse all .ttl files (or load pre-expanded sections if available)
     - Expand acronyms (skip if already expanded)
     - Concatenate all sections
     - Run NER across all sections at once
     - Split results back per paper
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_ner", debug=debug)

    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    
    expanded_dir = os.path.join(output_dir, "../sections")
    os.makedirs(expanded_dir, exist_ok=True)

    logger.info(f"✅ Starting NER for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Expanded sections dir: {expanded_dir}")
    
    if resume:
        processed = load_json(checkpoint_file, default={})
        logger.info(f"Resuming from checkpoint: {len(processed)} files already done")
    else:
        processed = {}
        logger.info(f"Processing all files (no files processed or resume=False)")

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]
    remaining_files = [f for f in all_files if f not in processed]
    total = len(remaining_files)
    logger.info(f"Found {len(all_files)} files ({total} pending)")
    
    files_with_expanded = [
        f for f in remaining_files 
        if os.path.exists(get_expanded_csv_path(f, expanded_dir))
    ]

    if len(files_with_expanded) > 0:
        logger.info(f"📂 {len(files_with_expanded)}/{total} pending files already have expanded sections (will skip expansion)")

    # Initialize gazetteer if enabled
    gazetteer = None
    domain_conf = DOMAIN_MODELS.get(domain)
    if domain_conf.get('gazetteer', {}).get('enabled'):
        gaz_conf = domain_conf['gazetteer']
        gazetteer = GazetteerLinker(
            taxonomy_path=gaz_conf['taxonomy_path'],
            taxonomy_source=gaz_conf.get('taxonomy_source'),
            model_name=gaz_conf.get('model_name'),
            default_type=gaz_conf.get('default_type'),
            domain=domain,
            min_term_length=gaz_conf.get('min_term_length', 2),
            blocked_terms=gaz_conf.get('blocked_terms'),
            logger=logger
        )
        logger.info(f"✅ Gazetteer loaded: {gazetteer.model_name} from {gaz_conf['taxonomy_path']}")
        if gaz_conf.get('blocked_terms'):
            logger.info(f"   Blocked terms: {len(gaz_conf['blocked_terms'])} terms")

    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        logger.info(f"\n🧩 Processing batch {start // file_batch_size + 1} "
                    f"({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}
        
        files_expanded_from_cache = 0
        files_newly_expanded = 0
        files_empty = 0
        files_error = 0

        for path in tqdm(batch_files, desc=f"Batch {start // file_batch_size + 1}"):
            try:
                expanded_csv_path = get_expanded_csv_path(path, expanded_dir)
                
                if os.path.exists(expanded_csv_path):
                    df = load_expanded_sections(expanded_csv_path, logger=logger)
                    if df is None or df.empty:
                        processed[path] = {"status": "empty_expanded"}
                        files_empty += 1
                        continue
                    
                    df["paper_path"] = path
                    for sid in df["section_id"].tolist():
                        paper_map[sid] = path
                    all_sections.append(df)
                    files_expanded_from_cache += 1
                    continue
                
                logger.info(f"📄 No pre-expanded sections for {path}, re-expanding...")
                
                records = parse_nif_file(path, logger=logger)
                if not records:
                    processed[path] = {"status": "empty"}
                    files_empty += 1
                    continue

                df = pd.DataFrame(records)
                df = apply_acronym_expansion(df, logger=logger)
                df["paper_path"] = path

                for sid in df["section_id"].tolist():
                    paper_map[sid] = path

                all_sections.append(df)
                files_newly_expanded += 1
                
                df.to_csv(expanded_csv_path, index=False)
                
            except Exception as e:
                logger.error(f"⚠️ Error parsing {path}: {e}")
                processed[path] = {"status": f"parse_error: {e}"}
                files_error += 1

        logger.info(f"📊 Batch expansion stats: "
                    f"{files_expanded_from_cache} from cache, "
                    f"{files_newly_expanded} newly expanded, "
                    f"{files_empty} empty, "
                    f"{files_error} errors")

        if not all_sections:
            logger.warning("⚠️ No sections to process in this batch.")
            continue

        df_all = pd.concat(all_sections, ignore_index=True).fillna("")
        logger.info(f"📊 Combined batch: {len(df_all)} sections total")
        
        for path, group in df_all.groupby("paper_path"):
            expanded_csv_path = get_expanded_csv_path(path, expanded_dir)
            if not os.path.exists(expanded_csv_path):
                group.to_csv(expanded_csv_path, index=False)

        # Run gazetteer
        df_gazetteer = pd.DataFrame()
        if gazetteer:
            logger.info("🔍 Running gazetteer-based entity extraction...")
            gazetteer_entities = []
            for _, row in df_all.iterrows():
                gaz_ents = gazetteer.extract_entities(
                    text=row['section_content_expanded'],
                    section_id=row['section_id'],
                    domain=domain
                )
                gazetteer_entities.append({
                    'section_id': row['section_id'],
                    'entities': gaz_ents
                })
            df_gazetteer = pd.DataFrame(gazetteer_entities)
            total_gaz = sum(len(e['entities']) for e in gazetteer_entities)
            logger.info(f"✅ Gazetteer found {total_gaz} entities")

        # Run NER models (unless gazetteer_only)
        df_ner = pd.DataFrame()
        if not gazetter_only:
            logger.info("🤖 Running NER models...")
            df_ner = predict_sections_multimodel(
                df_all,
                domain=domain,
                logger=logger
            )
            total_ner = sum(len(e) for e in df_ner['entities'].tolist())
            logger.info(f"✅ NER found {total_ner} entities")

        # Merge gazetteer + NER
        if not df_gazetteer.empty and not df_ner.empty:
            df_merged = merge_gazetteer_and_ner(df_gazetteer, df_ner, logger)
        elif not df_gazetteer.empty:
            df_merged = df_gazetteer
        elif not df_ner.empty:
            df_merged = df_ner
        else:
            logger.warning("⚠️ No entities found in this batch")
            continue

        # Split and save per paper
        logger.info("💾 Saving results per paper...")
        for path in batch_files:
            paper_section_ids = [sid for sid, p in paper_map.items() if p == path]
            paper_df = df_merged[df_merged['section_id'].isin(paper_section_ids)]

            if paper_df.empty:
                processed[path] = {"status": "no_entities"}
                continue

            out_name = os.path.basename(path).replace(".ttl", ".jsonl")
            out_path = os.path.join(output_dir, out_name)
            
            paper_df.to_json(out_path, orient='records', lines=True, force_ascii=False)

            total_ents = sum(len(e) for e in paper_df['entities'].tolist())
            processed[path] = {"status": "done", "entities": total_ents}

        save_json(processed, checkpoint_file)
        logger.info(f"💾 Checkpoint saved: {len(processed)} files processed")

    logger.info("🎉 All batches processed successfully.")


# ============================================================
# Entity Linking Function
# ============================================================

def run_el(
    domain: str,
    ner_output_dir: str,
    el_output_dir: str,
    linker_type: str = "auto",
    model_name: str = "intfloat/multilingual-e5-base",
    taxonomy_path: str = "taxonomies/energy/IRENA.tsv",
    taxonomy_source: str = "IRENA",
    threshold: float = 0.7,
    context_window: int = 3,
    max_contexts: int = 3,
    use_sentence_context: bool = False,
    use_context_for_retrieval: bool = False,
    reranker_llm: str = "Qwen/Qwen3-1.7B",
    reranker_top_k: int = 5,
    reranker_fallbacks: bool = True,
    reranker_thinking: bool = False,
    resume: bool = False,
    debug: bool = False
):
    """
    Run Entity Linking on NER outputs.
    
    Supports multiple linking strategies based on domain configuration:
    - "semantic": SemanticLinker (default for most domains)
    - "reranker": RerankerLinker with LLM
    - "fts5": FTS5Linker with per-entity-type indices
    
    Now supports domain-level entity filters:
    - blocked_mentions: Skip specific terms
    - min_mention_length: Skip short mentions
    """
    os.makedirs(el_output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(el_output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(el_output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_el", debug=debug)
    
    # Get domain configuration
    domain_conf = DOMAIN_MODELS.get(domain, {})
    linking_strategy = domain_conf.get("linking_strategy", "semantic")
    
    logger.info(f"🔗 Starting Entity Linking for domain={domain}")
    logger.info(f"NER input: {ner_output_dir}")
    logger.info(f"EL output: {el_output_dir}")
    logger.info(f"Linking strategy: {linking_strategy}")

    # Load entity filters (blocked mentions + min length)
    entity_filters = load_entity_filters(domain_conf, logger)
    total_blocked_terms = sum(len(m) for m in entity_filters["blocked_mentions"].values())
    default_min_len = entity_filters["min_mention_length"].get("_default", 1)
    if total_blocked_terms > 0 or default_min_len > 1:
        logger.info(f"🔧 Entity filters active: {total_blocked_terms} blocked terms, min_length={default_min_len}")

    # Load checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    
    if resume:
        processed = load_json(checkpoint_file, default={})
        logger.info(f"📦 Checkpoint loaded: {len(processed)} files already processed")
    else:
        processed = {}
        logger.info(f"Processing all files (no files processed or resume=False)")
    
    # Load cache
    cache_dir = os.path.join(el_output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "linking_cache.json")
    cache = load_json(cache_file, default={})
    logger.info(f"💾 Cache loaded: {len(cache)} entries")

    # === Initialize linkers based on strategy ===
    
    if linking_strategy == "fts5":
        from src.fts5_linker import FTS5Linker
        
        fts5_config = domain_conf.get("fts5_linkers", {})
        fts5_linkers = {}
        
        logger.info("Initializing FTS5 linkers:")
        for entity_type, config in fts5_config.items():
            try:
                fts5_linkers[entity_type.lower()] = {
                    "linker": FTS5Linker(
                        index_path=config["index_path"],
                        taxonomy_source=config["taxonomy_source"],
                        logger=logger
                    ),
                    "fallback": config.get("fallback"),
                }
                logger.info(f"  ✅ {entity_type}: {config['index_path']}")
            except Exception as e:
                logger.error(f"  ❌ {entity_type}: {e}")
        
        # Initialize semantic fallback linker if needed
        semantic_fallback_linker = None
        semantic_fallback_config = domain_conf.get("semantic_fallback", {})
        
        if semantic_fallback_config:
            needs_fallback = any(
                cfg.get("fallback") == "semantic" 
                for cfg in fts5_config.values()
            )
            
            if needs_fallback:
                from src.semantic_linker import SemanticLinker
                
                fallback_type = next(
                    (t for t, cfg in fts5_config.items() if cfg.get("fallback") == "semantic"),
                    None
                )
                
                if fallback_type and fallback_type in semantic_fallback_config:
                    fb_config = semantic_fallback_config[fallback_type]
                    logger.info(f"Initializing semantic fallback for {fallback_type}:")
                    logger.info(f"  Taxonomy: {fb_config['taxonomy_path']}")
                    
                    semantic_fallback_linker = SemanticLinker(
                        model_name=fb_config.get("model_name", model_name),
                        taxonomy_path=fb_config["taxonomy_path"],
                        threshold=fb_config.get("threshold", threshold),
                        context_window=context_window,
                        max_contexts=max_contexts,
                        use_sentence_context=use_sentence_context,
                        logger=logger
                    )
                    logger.info(f"  ✅ Semantic fallback ready")
        
        linker = None
        
    else:
        # Traditional linking (semantic/instruct/reranker)
        fts5_linkers = None
        semantic_fallback_linker = None
        
        if linker_type == "auto":
            if "qwen" in model_name.lower() or "qwen" in reranker_llm.lower():
                linker_type = "reranker"
            elif "instruct" in model_name.lower():
                linker_type = "instruct"
            else:
                linker_type = "semantic"
        
        logger.info(f"Linker type: {linker_type}")
        logger.info(f"Taxonomy: {taxonomy_path} (source: {taxonomy_source})")
        logger.info(f"Threshold: {threshold}")
        logger.info(f"Context window: {context_window} {'tokens' if context_window > 0 else '(disabled)'}")
        logger.info(f"Max. contexts: {max_contexts}")
        if context_window > 0:
            logger.info(f"Context type: {'sentences' if use_sentence_context else 'token windows'}")

        if linker_type == "reranker":
            from src.reranker_linker import RerankerLinker
            
            logger.info(f"Reranker configuration:")
            logger.info(f"  Embedding model: {model_name}")
            logger.info(f"  LLM model: {reranker_llm}")
            logger.info(f"  Top-k candidates: {reranker_top_k}")
            logger.info(f"  Context for retrieval: {use_context_for_retrieval}")
            logger.info(f"  Top-level fallbacks: {reranker_fallbacks}")
            logger.info(f"  Thinking mode: {reranker_thinking}")
            
            linker = RerankerLinker(
                taxonomy_path=taxonomy_path,
                domain=domain,
                embedding_model_name=model_name,
                llm_model_name=reranker_llm,
                threshold=threshold,
                context_window=context_window,
                max_contexts=max_contexts,
                use_sentence_context=use_sentence_context,
                use_context_for_retrieval=use_context_for_retrieval,
                top_k_candidates=reranker_top_k,
                add_top_level_fallbacks=reranker_fallbacks,
                enable_thinking=reranker_thinking,
                logger=logger
            )
        
        elif linker_type == "instruct":
            from src.instruct_linker import InstructLinker
            
            logger.info(f"Instruct model: {model_name}")
            
            linker = InstructLinker(
                model_name=model_name,
                taxonomy_path=taxonomy_path,
                threshold=threshold,
                context_window=context_window,
                max_contexts=max_contexts,
                use_sentence_context=use_sentence_context,
                logger=logger
            )
        
        else:  # semantic
            from src.semantic_linker import SemanticLinker
            
            logger.info(f"Semantic model: {model_name}")
            
            linker = SemanticLinker(
                model_name=model_name,
                taxonomy_path=taxonomy_path,
                threshold=threshold,
                context_window=context_window,
                max_contexts=max_contexts,
                use_sentence_context=use_sentence_context,
                logger=logger
            )
    
    # Find NER output files
    ner_files = [
        os.path.join(ner_output_dir, f) 
        for f in os.listdir(ner_output_dir) 
        if f.endswith('.jsonl')
    ]
    
    remaining_files = [f for f in ner_files if f not in processed]
    logger.info(f"Found {len(ner_files)} NER output files ({len(remaining_files)} pending)")
    
    # Get sections directory
    sections_dir = os.path.join(os.path.dirname(ner_output_dir), "sections")
    
    # Process files
    files_processed = 0
    
    for ner_file in tqdm(remaining_files, desc="Entity Linking"):
        try:
            # Load NER output
            df_ner = pd.read_json(ner_file, lines=True)
            if df_ner.empty:
                processed[ner_file] = {"status": "empty"}
                save_json(processed, checkpoint_file)
                continue
            
            # Load expanded sections for context
            sections_file = os.path.basename(ner_file).replace(".jsonl", "_sections.csv")
            sections_path = os.path.join(sections_dir, sections_file)
            
            if not os.path.exists(sections_path):
                sections_file = os.path.basename(ner_file).replace(".jsonl", ".csv")
                sections_path = os.path.join(sections_dir, sections_file)
            
            if not os.path.exists(sections_path):
                logger.warning(f"⚠️ No sections file for {ner_file}, skipping context")
                text_map = {}
            else:
                try:
                    df_expanded = pd.read_csv(sections_path, dtype=str).fillna("")
                    text_map = dict(zip(
                        df_expanded['section_id'],
                        df_expanded['section_content_expanded']
                    ))
                except Exception as csv_error:
                    logger.error(f"❌ Failed to load CSV: {csv_error}")
                    logger.error(traceback.format_exc())
                    processed[ner_file] = {"status": "csv_load_error", "error": str(csv_error)[:200]}
                    save_json(processed, checkpoint_file)
                    continue
            
            # Link entities per section
            linked_sections = []
            total_entities = 0
            total_linked = 0
            total_blocked = 0
            total_too_short = 0
            
            for _, row in df_ner.iterrows():
                section_id = row['section_id']
                entities = row['entities']
                section_text = text_map.get(section_id, "")
                
                if not section_text:
                    linked_sections.append(row.to_dict())
                    continue
                
                # Filter entities (blocked mentions + min length)
                filtered_entities, filter_stats = filter_entities(
                    entities, 
                    entity_filters, 
                    logger=logger,
                    log_skipped=debug
                )
                total_blocked += filter_stats.get("blocked", 0)
                total_too_short += filter_stats.get("too_short", 0)
                
                # Link entities based on strategy
                if linking_strategy == "fts5" and fts5_linkers:
                    enriched_entities = []
                    
                    for entity in filtered_entities:
                        if entity.get('linking'):
                            enriched_entities.append(entity)
                            total_linked += 1
                            continue
                        
                        entity_type = entity.get('entity', '').lower()
                        entity_text = entity.get('text', '')
                        entity_text_lower = entity_text.lower()
                        
                        if entity_type not in fts5_linkers:
                            enriched_entities.append(entity)
                            continue
                        
                        linker_config = fts5_linkers[entity_type]
                        fts5_linker = linker_config["linker"]
                        fallback = linker_config.get("fallback")
                        
                        cache_key = f"{entity_type}:{entity_text_lower}"
                        if cache_key in cache:
                            cached = cache[cache_key]
                            entity_copy = entity.copy()
                            if cached.get('linking'):
                                entity_copy['linking'] = cached['linking']
                                total_linked += 1
                            enriched_entities.append(entity_copy)
                            continue
                        
                        linking = fts5_linker.link_entity(entity_text)
                        
                        if not linking and fallback == "semantic" and semantic_fallback_linker:
                            fallback_source = semantic_fallback_config.get(entity_type, {}).get("taxonomy_source", "DOID")
                            linking = semantic_fallback_linker.link_entity_all_contexts(
                                entity_text,
                                [section_text[:500]],
                                fallback_source
                            )
                        
                        cache[cache_key] = {'linking': linking}
                        
                        entity_copy = entity.copy()
                        if linking:
                            entity_copy['linking'] = linking
                            total_linked += 1
                        enriched_entities.append(entity_copy)
                    
                    total_entities += len(filtered_entities)
                    
                else:
                    # Traditional linking (semantic/instruct/reranker)
                    enriched_entities, cache = linker.link_entities_in_section(
                        section_text, 
                        filtered_entities,
                        cache, 
                        taxonomy_source,
                        filename=os.path.basename(ner_file),
                        section_id=section_id
                    )
                    
                    total_entities += len(filtered_entities)
                    total_linked += sum(1 for e in enriched_entities if e.get('linking'))
                
                linked_sections.append({
                    'section_id': section_id,
                    'entities': enriched_entities
                })
            
            # Save linked output
            output_file = os.path.join(
                el_output_dir,
                os.path.basename(ner_file)
            )
            df_linked = pd.DataFrame(linked_sections)
            df_linked.to_json(output_file, orient='records', lines=True, force_ascii=False)
            
            linking_rate = f"{100*total_linked/total_entities:.1f}%" if total_entities > 0 else "0%"
            total_filtered = total_blocked + total_too_short
            
            processed[ner_file] = {
                "status": "done",
                "total_entities": total_entities,
                "linked_entities": total_linked,
                "filtered_entities": total_filtered,
                "linking_rate": linking_rate
            }
            
            filter_msg = ""
            if total_filtered > 0:
                parts = []
                if total_blocked > 0:
                    parts.append(f"{total_blocked} blocked")
                if total_too_short > 0:
                    parts.append(f"{total_too_short} too short")
                filter_msg = f", {total_filtered} filtered ({', '.join(parts)})"
            
            logger.info(
                f"✅ {os.path.basename(ner_file)}: "
                f"{total_linked}/{total_entities} linked ({linking_rate})"
                f"{filter_msg}"
            )
            
            save_json(processed, checkpoint_file)

            log_memory_usage(logger, f"after file {files_processed + 1}")
            files_processed += 1
            
            if files_processed % 100 == 0:
                save_json(cache, cache_file)
                logger.info(f"💾 Cache checkpoint saved: {len(cache)} entries")
                    
            if files_processed % 500 == 0:
                log_cache_stats(cache, logger)
            
        except Exception as e:
            logger.error(f"❌ Error processing {ner_file}: {e}")
            logger.debug(traceback.format_exc())
            processed[ner_file] = {"status": f"error: {e}"}
            save_json(processed, checkpoint_file)
            continue
    
    # Save checkpoint and cache
    save_json(processed, checkpoint_file)
    save_json(cache, cache_file)
    
    logger.info(f"🎉 Entity Linking complete!")
    logger.info(f"💾 Final cache size: {len(cache)} entries")
    
    # Print summary statistics
    total_files = len([p for p in processed.values() if p.get('status') == 'done'])
    total_ents = sum(p.get('total_entities', 0) for p in processed.values() if p.get('status') == 'done')
    total_links = sum(p.get('linked_entities', 0) for p in processed.values() if p.get('status') == 'done')
    total_filt = sum(p.get('filtered_entities', 0) for p in processed.values() if p.get('status') == 'done')
    
    logger.info(f"📊 Summary: {total_files} files, {total_ents} entities, {total_links} linked")
    if total_filt > 0:
        logger.info(f"   Filtered (blocked/too short): {total_filt}")
    if total_ents > 0:
        logger.info(f"   Overall linking rate: {100*total_links/total_ents:.1f}%")


# ============================================================
# Geotagging Function
# ============================================================

def run_geotagging(domain, input_dir, output_dir, resume=True, file_batch_size=100, device="cpu"):
    """
    Run Geotagging pipeline (GeoNER + Role Classification).
    """
    from src.geotagging_runner import run_geotagging_batch

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_geotagging")

    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    processed = load_json(checkpoint_file, default={})

    logger.info(f"🌍 Starting Geotagging for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Resuming from checkpoint: {len(processed)} files already done")

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]
    remaining_files = [f for f in all_files if f not in processed]
    total = len(remaining_files)
    logger.info(f"Found {len(all_files)} files ({total} pending)")

    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        batch_idx = start // file_batch_size + 1
        logger.info(f"\n🧩 Processing batch {batch_idx} ({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}

        for path in tqdm(batch_files, desc=f"Batch {batch_idx}"):
            try:
                records = parse_nif_file(path, logger=logger)
                if not records:
                    processed[path] = {"status": "empty"}
                    continue

                df = pd.DataFrame(records)
                df = apply_acronym_expansion(df, logger=logger)
                df["paper_path"] = path

                for sid in df["section_id"].tolist():
                    paper_map[sid] = path

                all_sections.append(df)
            except Exception as e:
                logger.error(f"⚠️ Error parsing {path}: {e}")
                processed[path] = {"status": f"parse_error: {e}"}

        if not all_sections:
            logger.warning("⚠️ No sections to process in this batch.")
            continue

        df_all = pd.concat(all_sections, ignore_index=True).fillna("")
        logger.info(f"📊 Batch sections: {len(df_all)}")

        df_enriched = run_geotagging_batch(
            df_all,
            device=device,
            logger=logger
        )

        for path in batch_files:
            paper_section_ids = [sid for sid, p in paper_map.items() if p == path]
            paper_df = df_enriched[df_enriched['section_id'].isin(paper_section_ids)]

            if paper_df.empty:
                processed[path] = {"status": "no_entities"}
                continue

            out_name = os.path.basename(path).replace(".ttl", ".jsonl")
            out_path = os.path.join(output_dir, out_name)
            paper_df.to_json(out_path, orient='records', lines=True, force_ascii=False)

            total_ents = sum(len(e) for e in paper_df['entities'].tolist())
            processed[path] = {"status": "done", "entities": total_ents}

        save_json(processed, checkpoint_file)
        logger.info(f"💾 Checkpoint saved: {len(processed)} files processed")

    logger.info("🎉 Geotagging complete.")


# ============================================================
# Affiliations Function
# ============================================================

def run_affiliations(domain, input_dir, output_dir, resume=True, file_batch_size=100, device="cpu"):
    """
    Run AffilGood affiliation extraction and normalization.
    """
    from rdflib import Graph
    import re
    from src.affilgood_runner import run_affilgood_batch, AffilGood

    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_affiliations")

    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    if resume:
        processed = load_json(checkpoint_file, default={})
    else:
        processed = {}

    logger.info(f"🏛️ Starting Affiliation extraction for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Resuming from checkpoint: {len(processed)} affiliations already done")

    affil_good = AffilGood(device=device)
    logger.info("✅ AffilGood initialized")

    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]

    logger.info(f"Found {len(all_files)} TTL files")

    seen_affiliations = set(processed.keys())

    for path in tqdm(all_files, desc="Extracting + processing affiliations"):
        try:
            g = Graph()
            g.parse(path, format="turtle")

            affiliations = []
            for subj, pred, obj in g.triples((None, None, None)):
                if "affRawAffiliationString" in str(pred):
                    aff = str(obj)
                    aff = re.sub(r"[\n\t]+", ", ", aff)
                    aff = re.sub(r"\s*,\s*", ", ", aff)
                    aff = re.sub(r"\s{2,}", " ", aff)
                    aff = aff.strip(" ,;")

                    aff_id = str(subj)
                    if resume and aff_id in seen_affiliations:
                        continue

                    affiliations.append({
                        "affiliation_id": aff_id,
                        "raw_affiliation": aff,
                        "source_file": os.path.basename(path),
                    })
                    seen_affiliations.add(aff_id)

            if not affiliations:
                logger.info(f"⚪ {os.path.basename(path)} — no new affiliations found")
                continue

            logger.info(f"📄 {os.path.basename(path)} — {len(affiliations)} new affiliations found")

            df = pd.DataFrame(affiliations)
            base_name = os.path.basename(path).replace(".ttl", "_affilgood.jsonl")
            out_path = os.path.join(output_dir, base_name)

            results = run_affilgood_batch(
                df_batch=df,
                affilgood=affil_good,
                logger=logger,
                batch_size=file_batch_size,
                output_path=out_path,
            )

            for r in results:
                processed[r["affiliation_id"]] = {
                    "status": "done",
                    "source_file": r["raw_affiliation"],
                }

            save_json(processed, checkpoint_file)
            logger.info(f"💾 Saved checkpoint ({len(processed)} total processed)")

        except Exception as e:
            logger.error(f"⚠️ Error processing {path}: {e}")

    logger.info("🎉 Finished extracting and enriching all affiliations with AffilGood.")


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SciLake NER & Entity Linking Pipeline")
    parser.add_argument("--domain", required=True, help="Domain name (ccam, energy, etc.)")
    parser.add_argument("--input", help="Path to input directory or file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--step", default="all", help="gaz | ner | el | geotagging | affiliations | all")
    
    # NEW: Input format argument
    parser.add_argument("--input_format", default="nif", choices=["nif", "title_abstract", "legal_text"],
                       help="Input format: nif (TTL files), title_abstract (JSON), or legal_text (JSON)")
    
    # Entity Linking arguments
    parser.add_argument("--threshold", type=float, default=0.7, help="EL similarity threshold")
    parser.add_argument("--use_sentence_context", action="store_true", help="Use sentences instead of token windows")
    parser.add_argument("--use_context_for_retrieval", action="store_true",
                       help="Use context in embedding retrieval (stage 1). LLM (stage 2) always uses context.")
    parser.add_argument("--max_contexts", type=int, default=3, help="Max number of contexts to extract")
    parser.add_argument("--context_window", type=int, default=3, help="EL similarity context window")
    parser.add_argument("--el_model_name", default="intfloat/multilingual-e5-base", help="Model for entity linking")
    parser.add_argument("--taxonomy", default="taxonomies/energy/IRENA.tsv", help="Path to taxonomy TSV")
    parser.add_argument("--taxonomy_source", default="IRENA", help="Taxonomy source name (e.g., IRENA, UBERON)")
    
    # Linker type arguments
    parser.add_argument("--linker_type", default="auto", 
                       help="Linker type: auto | semantic | instruct | reranker")
    
    # Reranker-specific arguments
    parser.add_argument("--reranker_llm", default="Qwen/Qwen3-1.7B", 
                       help="LLM model for reranker linker")
    parser.add_argument("--reranker_top_k", type=int, default=5, 
                       help="Number of candidates for reranker")
    parser.add_argument("--reranker_fallbacks", action="store_true", default=True,
                       help="Add top-level fallbacks in reranker")
    parser.add_argument("--reranker_thinking", action="store_true", 
                       help="Enable LLM thinking mode (slower but more accurate)")
    
    # Other arguments
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=1000, help="Sections/files per batch")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device set to: {device}")

    # Route to appropriate function based on step and input format
    if args.step == "gaz":
        if not args.input:
            print("❌ Error: --input required for gazetteer step")
            return
        
        if args.input_format == "title_abstract":
            run_ner_title_abstract(
                domain=args.domain,
                input_path=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                batch_size=args.batch_size,
                debug=args.debug,
                gazetteer_only=True
            )
        else:
            run_ner(
                domain=args.domain,
                input_dir=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                file_batch_size=args.batch_size,
                debug=args.debug,
                gazetter_only=True
            )

    elif args.step == "ner":
        if not args.input:
            print("❌ Error: --input required for NER step")
            return
        
        if args.input_format == "title_abstract":
            run_ner_title_abstract(
                domain=args.domain,
                input_path=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                batch_size=args.batch_size,
                debug=args.debug,
                gazetteer_only=False
            )
        elif args.input_format == "legal_text":
            run_ner_legal_text(
                domain=args.domain,
                input_path=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                batch_size=args.batch_size,
                debug=args.debug,
                gazetteer_only=False
            )
        else:
            run_ner(
                domain=args.domain,
                input_dir=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                file_batch_size=args.batch_size,
                debug=args.debug,
            )
        
    elif args.step == "geotagging":
        if not args.input:
            print("❌ Error: --input required for geotagging step")
            return
        run_geotagging(
            domain=args.domain,
            input_dir=args.input,
            output_dir=os.path.join(args.output, "geotagging-ner"),
            resume=args.resume,
            file_batch_size=args.batch_size,
            device=device,  
        )
        
    elif args.step == "affiliations":
        if not args.input:
            print("❌ Error: --input required for affiliations step")
            return
        run_affiliations(
            domain=args.domain,
            input_dir=args.input,
            output_dir=os.path.join(args.output, "affiliations"),
            resume=args.resume,
            file_batch_size=args.batch_size,
            device=device,
        )

    elif args.step == "el":
        run_el(
            domain=args.domain,
            model_name=args.el_model_name,
            ner_output_dir=os.path.join(args.output, "ner"),
            el_output_dir=os.path.join(args.output, "el"),
            taxonomy_path=args.taxonomy,
            taxonomy_source=args.taxonomy_source,
            use_sentence_context=args.use_sentence_context,
            use_context_for_retrieval=args.use_context_for_retrieval,
            threshold=args.threshold,
            context_window=args.context_window,
            max_contexts=args.max_contexts,
            linker_type=args.linker_type,
            reranker_llm=args.reranker_llm,
            reranker_top_k=args.reranker_top_k,
            reranker_fallbacks=args.reranker_fallbacks,
            reranker_thinking=args.reranker_thinking,
            resume=args.resume,
            debug=args.debug
        )
    
    elif args.step == "all":
        if not args.input:
            print("❌ Error: --input required for 'all' step")
            return
        
        # Run NER first
        print("🔹 Step 1/2: Running NER...")
        if args.input_format == "title_abstract":
            run_ner_title_abstract(
                domain=args.domain,
                input_path=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                batch_size=args.batch_size,
                debug=args.debug,
                gazetteer_only=False
            )
        else:
            run_ner(
                domain=args.domain,
                input_dir=args.input,
                output_dir=os.path.join(args.output, "ner"),
                resume=args.resume,
                file_batch_size=args.batch_size,
                debug=args.debug,
            )
        
        # Then run EL
        print("\n🔹 Step 2/2: Running Entity Linking...")
        run_el(
            domain=args.domain,
            model_name=args.el_model_name,
            ner_output_dir=os.path.join(args.output, "ner"),
            el_output_dir=os.path.join(args.output, "el"),
            taxonomy_path=args.taxonomy,
            taxonomy_source=args.taxonomy_source,
            use_sentence_context=args.use_sentence_context,
            use_context_for_retrieval=args.use_context_for_retrieval,
            threshold=args.threshold,
            context_window=args.context_window,
            max_contexts=args.max_contexts,
            linker_type=args.linker_type,
            reranker_llm=args.reranker_llm,
            reranker_top_k=args.reranker_top_k,
            reranker_fallbacks=args.reranker_fallbacks,
            reranker_thinking=args.reranker_thinking,
            resume=args.resume,
            debug=args.debug
        )
    
    else:
        print(f"❌ Unknown step: {args.step}. Use: gaz | ner | el | geotagging | affiliations | all")


if __name__ == "__main__":
    main()
