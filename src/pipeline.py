import argparse
import os
import pandas as pd
import traceback
import resource

from tqdm import tqdm
import torch

from src.utils.logger import setup_logger
from src.utils.io_utils import load_json, save_json, append_jsonl
from src.nif_reader import parse_nif_file, apply_acronym_expansion
from src.ner_runner import predict_sections_multimodel
from src.gazetteer_linker import GazetteerLinker
from configs.domain_models import DOMAIN_MODELS


def log_memory_usage(logger, label=""):
    """Log current memory usage (Linux only)"""
    try:
        # Memory in MB
        mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        logger.info(f"📊 Memory {label}: {mem_mb:.2f} MB")
        
        # GPU memory if available
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
        
        # Remove NER entities that overlap with gazetteer (gazetteer has priority)
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
    
    # Directory for expanded sections (separate from NER checkpoint)
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

    # --- find input files ---
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

    # --- batch iteration ---
    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        logger.info(f"\n🧩 Processing batch {start // file_batch_size + 1} "
                    f"({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}  # section_id → paper_path
        
        # Counters for logging
        files_expanded_from_cache = 0
        files_newly_expanded = 0
        files_empty = 0
        files_error = 0

        # Parse all papers in the batch
        for path in tqdm(batch_files, desc=f"Batch {start // file_batch_size + 1}"):
            try:
                # Check if expanded sections already exist
                expanded_csv_path = get_expanded_csv_path(path, expanded_dir)
                
                if os.path.exists(expanded_csv_path):
                    # Load pre-expanded sections
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
                
                # Parse and expand (no pre-expanded sections available)
                records = parse_nif_file(path, logger=logger)
                if not records:
                    processed[path] = {"status": "empty"}
                    files_empty += 1
                    continue

                df = pd.DataFrame(records)
                df = apply_acronym_expansion(df, logger=logger)
                df["paper_path"] = path

                # Keep mapping to rebuild per-paper outputs later
                for sid in df["section_id"].tolist():
                    paper_map[sid] = path

                all_sections.append(df)
                files_newly_expanded += 1
                
                # Save expanded sections for this file (for future runs)
                df.to_csv(expanded_csv_path, index=False)
                
            except Exception as e:
                logger.error(f"⚠️ Error parsing {path}: {e}")
                processed[path] = {"status": f"parse_error: {e}"}
                files_error += 1

        # Log expansion statistics for this batch
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
        
        # Save any newly expanded sections that weren't saved yet
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

        # --- Split and save per paper ---
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

        # Save checkpoint
        save_json(processed, checkpoint_file)
        logger.info(f"💾 Checkpoint saved: {len(processed)} files processed")

    logger.info("🎉 All batches processed successfully.")


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
    # Reranker-specific parameters
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
    - "fts5": FTS5Linker with per-entity-type indices (e.g., cancer domain)
    
    Args:
        domain: Domain name
        ner_output_dir: Directory with NER .jsonl outputs
        el_output_dir: Output directory for linked entities
        linker_type: Type of linker (auto | semantic | instruct | reranker)
        model_name: Embedding model name (used for semantic/instruct/reranker retrieval)
        taxonomy_path: Path to taxonomy TSV file
        taxonomy_source: Name of taxonomy (e.g., "IRENA", "UBERON")
        threshold: Similarity threshold for embedding retrieval
        context_window: Number of tokens to consider in context window (0 = no context)
        max_contexts: Max number of contexts to extract
        use_sentence_context: Use full sentences instead of token windows
        use_context_for_retrieval: Use context in embedding retrieval (default: False)
        reranker_llm: LLM model for reranker linker
        reranker_top_k: Number of candidates for reranker
        reranker_fallbacks: Add top-level fallbacks in reranker
        reranker_thinking: Enable LLM thinking mode (slower but more accurate)
        resume: Resume from checkpoint
        debug: Enable debug logging
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
        # FTS5-based linking (e.g., cancer domain)
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
                    "blocked_mentions": config.get("blocked_mentions", set()),
                    "fallback": config.get("fallback"),
                }
                logger.info(f"  ✅ {entity_type}: {config['index_path']}")
            except Exception as e:
                logger.error(f"  ❌ {entity_type}: {e}")
        
        # Initialize semantic fallback linker if needed
        semantic_fallback_linker = None
        semantic_fallback_config = domain_conf.get("semantic_fallback", {})
        
        if semantic_fallback_config:
            # Check if any entity type needs semantic fallback
            needs_fallback = any(
                cfg.get("fallback") == "semantic" 
                for cfg in fts5_config.values()
            )
            
            if needs_fallback:
                from src.semantic_linker import SemanticLinker
                
                # Use first fallback config (typically disease)
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
        
        linker = None  # Not using single linker for FTS5 strategy
        
    else:
        # Traditional linking (semantic/instruct/reranker)
        fts5_linkers = None
        semantic_fallback_linker = None
        
        # Auto-detect linker type from model name
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

        # Initialize linker based on type
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
    remaining = [f for f in ner_files if f not in processed]
    logger.info(f"📂 Found {len(ner_files)} NER files ({len(remaining)} pending)")
       
    # Load expanded sections directory (for text context)
    expanded_dir = os.path.join(os.path.dirname(ner_output_dir), "sections")
    
    if not os.path.exists(expanded_dir):
        logger.error(f"❌ Expanded directory not found: {expanded_dir}")
        logger.error("   Run NER step first to generate expanded sections")
        return

    files_processed = 0

    # Process each file
    for ner_file in tqdm(remaining, desc="Linking entities"):
        try:
            logger.info(f"🗃️  Processing file {files_processed + 1}/{len(remaining)}: {os.path.basename(ner_file)}")
        
            # Load NER entities
            df_ner = pd.read_json(ner_file, lines=True)
            
            # Load corresponding expanded text
            base_name = os.path.basename(ner_file).replace('.jsonl', '_sections.csv')
            expanded_file = os.path.join(expanded_dir, base_name)
            
            if not os.path.exists(expanded_file):
                logger.warning(f"⚠️  No expanded file for {ner_file}")
                processed[ner_file] = {"status": "no_expanded"}
                continue
                
            try:
                # Check file size first
                file_size = os.path.getsize(expanded_file)
                logger.info(f"  File size: {file_size / (1024*1024):.2f} MB")
            
                df_expanded = pd.read_csv(
                    expanded_file,
                    engine='python',
                    on_bad_lines='warn',
                    encoding='utf-8',
                    dtype=str
                )
                logger.info(f"  ✔ Loaded {len(df_expanded)} sections successfully")
                
            except Exception as csv_error:
                logger.error(f"❌ Failed to load CSV: {csv_error}")
                logger.error(traceback.format_exc())
                processed[ner_file] = {"status": "csv_load_error", "error": str(csv_error)[:200]}
                save_json(processed, checkpoint_file)
                continue
            
            # Create section_id -> text mapping
            text_map = dict(zip(
                df_expanded['section_id'],
                df_expanded['section_content_expanded']
            ))
            
            # Link entities per section
            linked_sections = []
            total_entities = 0
            total_linked = 0
            total_skipped = 0
            
            for _, row in df_ner.iterrows():
                section_id = row['section_id']
                entities = row['entities']
                section_text = text_map.get(section_id, "")
                
                if not section_text:
                    linked_sections.append(row.to_dict())
                    continue
                
                # === Link entities based on strategy ===
                
                if linking_strategy == "fts5" and fts5_linkers:
                    # FTS5-based linking (per entity type)
                    enriched_entities = []
                    
                    for entity in entities:
                        # Skip if already has linking (from gazetteer)
                        if entity.get('linking'):
                            enriched_entities.append(entity)
                            total_linked += 1
                            continue
                        
                        entity_type = entity.get('entity', '').lower()
                        entity_text = entity.get('text', '')
                        entity_text_lower = entity_text.lower()
                        
                        # Check if we have a linker for this entity type
                        if entity_type not in fts5_linkers:
                            # No linker for this type, keep as-is
                            enriched_entities.append(entity)
                            continue
                        
                        linker_config = fts5_linkers[entity_type]
                        fts5_linker = linker_config["linker"]
                        blocked_mentions = linker_config.get("blocked_mentions", set())
                        fallback = linker_config.get("fallback")
                        
                        # Check blocked mentions
                        if entity_text_lower in blocked_mentions:
                            logger.debug(f"Skipping blocked mention: '{entity_text}' ({entity_type})")
                            total_skipped += 1
                            # Don't add to enriched_entities - skip entirely
                            continue
                        
                        # Check cache
                        cache_key = f"{entity_type}:{entity_text_lower}"
                        if cache_key in cache:
                            cached = cache[cache_key]
                            entity_copy = entity.copy()
                            if cached.get('linking'):
                                entity_copy['linking'] = cached['linking']
                                total_linked += 1
                            enriched_entities.append(entity_copy)
                            continue
                        
                        # Try FTS5 linking
                        linking = fts5_linker.link_entity(entity_text)
                        
                        # Try semantic fallback if no match and fallback is configured
                        if not linking and fallback == "semantic" and semantic_fallback_linker:
                            # Use semantic linker for this entity
                            fallback_source = semantic_fallback_config.get(entity_type, {}).get("taxonomy_source", "DOID")
                            linking = semantic_fallback_linker.link_entity_all_contexts(
                                entity_text,
                                [section_text[:500]],  # Use section as context
                                fallback_source
                            )
                        
                        # Cache result
                        cache[cache_key] = {'linking': linking}
                        
                        # Add to entity
                        entity_copy = entity.copy()
                        if linking:
                            entity_copy['linking'] = linking
                            total_linked += 1
                        enriched_entities.append(entity_copy)
                    
                    total_entities += len(entities)
                    
                else:
                    # Traditional linking (semantic/instruct/reranker)
                    enriched_entities, cache = linker.link_entities_in_section(
                        section_text, 
                        entities, 
                        cache, 
                        taxonomy_source,
                        filename=os.path.basename(ner_file),
                        section_id=section_id
                    )
                    
                    total_entities += len(entities)
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
            
            processed[ner_file] = {
                "status": "done",
                "total_entities": total_entities,
                "linked_entities": total_linked,
                "skipped_entities": total_skipped,
                "linking_rate": linking_rate
            }
            
            logger.info(
                f"✅ {os.path.basename(ner_file)}: "
                f"{total_linked}/{total_entities} entities linked ({linking_rate})"
                + (f", {total_skipped} skipped" if total_skipped > 0 else "")
            )
            
            # Save checkpoint and cache
            save_json(processed, checkpoint_file)

            log_memory_usage(logger, f"after file {files_processed + 1}")
            files_processed += 1
            
            # Save cache every 100 files
            if files_processed % 100 == 0:
                save_json(cache, cache_file)
                if logger:
                    logger.info(f"💾 Cache checkpoint saved: {len(cache)} entries")
                    
            # After every 500 files save stats
            if files_processed % 500 == 0:
                log_cache_stats(cache, logger)
            
        except Exception as e:
            logger.error(f"❌ Error processing {ner_file}: {e}")
            logger.debug(traceback.format_exc())
            processed[ner_file] = {"status": f"error: {e}"}
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
    total_skip = sum(p.get('skipped_entities', 0) for p in processed.values() if p.get('status') == 'done')
    
    logger.info(f"📊 Summary: {total_files} files, {total_ents} entities, {total_links} linked")
    if total_skip > 0:
        logger.info(f"   Skipped (blocked): {total_skip}")
    if total_ents > 0:
        logger.info(f"   Overall linking rate: {100*total_links/total_ents:.1f}%")


def run_geotagging(domain, input_dir, output_dir, resume=True, file_batch_size=100, device="cpu"):
    """
    Run Geotagging pipeline (GeoNER + Role Classification).
    For each batch:
      - Parse all .ttl files
      - Expand acronyms
      - Concatenate all sections
      - Run GeoNER across all sections at once (batched inside the transformer)
      - Run Role Classification on detected mentions
      - Save enriched entities with ROLE per section
    """
    from src.geotagging_runner import run_geotagging_batch

    # --- Setup directories ---
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_geotagging")

    # --- Load checkpoint ---
    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    processed = load_json(checkpoint_file, default={})

    logger.info(f"🌍 Starting Geotagging for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Resuming from checkpoint: {len(processed)} files already done")

    # --- Find .ttl input files ---
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]
    remaining_files = [f for f in all_files if f not in processed]
    total = len(remaining_files)
    logger.info(f"Found {len(all_files)} files ({total} pending)")

    # --- Process in file batches ---
    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        batch_idx = start // file_batch_size + 1
        logger.info(f"\n🧩 Processing batch {batch_idx} ({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}

        # 1️⃣ Parse NIF and expand acronyms
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

        # 2️⃣ Run Geotagging
        df_enriched = run_geotagging_batch(
            df_all,
            device=device,
            logger=logger
        )

        # 3️⃣ Save per paper
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


def run_affiliations(domain, input_dir, output_dir, resume=True, file_batch_size=100, device="cpu"):
    """
    Run AffilGood affiliation extraction and normalization.
    """
    from rdflib import Graph
    import re
    from src.affilgood_runner import run_affilgood_batch, AffilGood

    # --- Setup directories ---
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    logger = setup_logger(log_dir, name=f"{domain}_affiliations")

    # --- Load checkpoint ---
    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    if resume:
        processed = load_json(checkpoint_file, default={})
    else:
        processed = {}

    logger.info(f"🏛️ Starting Affiliation extraction for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Resuming from checkpoint: {len(processed)} affiliations already done")

    # --- Initialize AffilGood ---
    affil_good = AffilGood(device=device)
    logger.info("✅ AffilGood initialized")

    # --- Find all .ttl input files ---
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]

    logger.info(f"Found {len(all_files)} TTL files")

    seen_affiliations = set(processed.keys())

    # --- Process files ---
    for path in tqdm(all_files, desc="Extracting + processing affiliations"):
        try:
            g = Graph()
            g.parse(path, format="turtle")

            affiliations = []
            for subj, pred, obj in g.triples((None, None, None)):
                if "affRawAffiliationString" in str(pred):
                    aff = str(obj)
                    # 🧹 Clean text
                    aff = re.sub(r"[\n\t]+", ", ", aff)
                    aff = re.sub(r"\s*,\s*", ", ", aff)
                    aff = re.sub(r"\s{2,}", " ", aff)
                    aff = aff.strip(" ,;")

                    aff_id = str(subj)
                    # Skip duplicates if resume mode
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

            # Convert to DataFrame for batch processing
            df = pd.DataFrame(affiliations)
            base_name = os.path.basename(path).replace(".ttl", "_affilgood.jsonl")
            out_path = os.path.join(output_dir, base_name)

            # 🔥 Run AffilGood enrichment
            results = run_affilgood_batch(
                df_batch=df,
                affilgood=affil_good,
                logger=logger,
                batch_size=file_batch_size,
                output_path=out_path,
            )

            # Update checkpoint
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


def main():
    parser = argparse.ArgumentParser(description="SciLake NER & Entity Linking Pipeline")
    parser.add_argument("--domain", required=True, help="Domain name (ccam, energy, etc.)")
    parser.add_argument("--input", help="Path to input NIF directory (for NER step)")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--step", default="all", help="gaz | ner | el | geotagging | affiliations | all")
    
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
    parser.add_argument("--batch_size", type=int, default=1000, help="Files per batch (for NER)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Device set to: {device}")

    if args.step == "gaz":
        if not args.input:
            print("❌ Error: --input required for gazetteer step")
            return
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
