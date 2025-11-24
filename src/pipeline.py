import argparse
import os
import pandas as pd
import traceback
import resource

from tqdm import tqdm

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

def run_ner(domain, input_dir, output_dir, resume=False, file_batch_size=100, debug=False, gazetter_only=False):
    """
    Run NER enrichment by batching multiple papers together.
    For each batch:
     - Parse all .ttl files
     - Expand acronyms
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

    logger.info(f"✅ Starting NER for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    
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
            domain=domain
        )
        logger.info(f"✅ Gazetteer loaded: {gazetteer.model_name} from {gaz_conf['taxonomy_path']}")

    # --- batch iteration ---
    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        logger.info(f"\n🧩 Processing batch {start // file_batch_size + 1} "
                    f"({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}  # section_id → paper_path

        # Parse all papers in the batch
        for path in tqdm(batch_files, desc=f"Batch {start // file_batch_size + 1}"):
            try:
                records = parse_nif_file(path, logger=logger)
                if not records:
                    processed[path] = {"status": "empty"}
                    continue

                df = pd.DataFrame(records)
                df = apply_acronym_expansion(df, logger=logger)
                df["paper_path"] = path

                # Keep mapping to rebuild per-paper outputs later
                for sid in df["section_id"].tolist():
                    paper_map[sid] = path

                all_sections.append(df)
            except Exception as e:
                logger.error(f"⚠️ Error parsing {path}: {e}")
                processed[path] = {"status": f"parse_error: {e}"}

        if not all_sections:
            logger.warning("⚠️ No sections to process in this batch.")
            continue

        df_all = pd.concat(all_sections, ignore_index=True)
        logger.info(f"📊 Combined batch: {len(df_all)} sections total")
        
        # Save expanded sections per input file for debugging
        expanded_dir = os.path.join(output_dir, "../sections")
        os.makedirs(expanded_dir, exist_ok=True)
        for path, group in df_all.groupby("paper_path"):
            base_name = os.path.basename(path).replace(".ttl", "_sections.csv")
            csv_path = os.path.join(expanded_dir, base_name)
            group.to_csv(csv_path, index=False)
        logger.info(f"Saved expanded sections to {expanded_dir}/")

        # Run gazeteer
        if gazetteer:
            logger.info("🔍 Running gazetteer-based entity extraction...")
            gazetteer_entities = []
            for _, row in df_all.iterrows():
                gaz_ents = gazetteer.extract_entities(
                    text=row['section_content_expanded'],
                    section_id=row['section_id'],
                    domain=domain
                )
                if gaz_ents:
                    gazetteer_entities.append({
                        'section_id': row['section_id'],
                        'entities': gaz_ents
                    })
            
            df_gazetteer = pd.DataFrame(gazetteer_entities) if gazetteer_entities else pd.DataFrame()

        if gazetter_only:
            df_entities = df_gazetteer
        else:
            # Run NER once for all sections in batch
            try:
                df_entities = predict_sections_multimodel(
                    df_all,
                    domain=domain,
                    text_col="section_content_expanded",
                    id_col="section_id",
                    stride=100,
                    batch_size=8,
                    logger=logger,
                )
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"🔥 NER failed on batch: {e}")
                logger.debug(tb)
                continue

            # Merge gazetteer + NER results
            if not df_gazetteer.empty:
                df_entities = merge_gazetteer_and_ner(df_gazetteer, df_entities, logger)

        # Group entities per paper and save
        section_dict = {}
        if not df_entities.empty:
            section_dict = dict(zip(df_entities["section_id"], df_entities["entities"]))
            logger.info(f"🧩 Grouping NER results per paper...")

        for paper_path, group_df in df_all.groupby("paper_path"):
            section_ids = group_df["section_id"].tolist()
            # Keep only sections with entities
            entities_subset = {
                sid: section_dict.get(sid, [])
                for sid in section_ids
                if section_dict.get(sid)
            }

            if not entities_subset:
                logger.info(f"⚪ No entities found for {paper_path} — skipping JSONL write.")
                processed[paper_path] = {"status": "no_entities"}
                continue

            # Write only non-empty entity sections
            base_name = os.path.basename(paper_path).replace(".ttl", ".jsonl")
            out_path = os.path.join(output_dir, base_name)
            for sid, ents in entities_subset.items():
                append_jsonl({"section_id": sid, "entities": ents}, out_path)

            processed[paper_path] = {"status": "done"}
            logger.info(f"✔️ Written {len(entities_subset)} sections for {paper_path}")

        # 4️⃣ Save checkpoint
        save_json(processed, checkpoint_file)
        logger.info(f"✅ Batch complete — total processed so far: {len(processed)} files.")

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
                                   LLM always uses context regardless of this setting
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
    
    # Auto-detect linker type from model name
    if linker_type == "auto":
        if "qwen" in model_name.lower() or "qwen" in reranker_llm.lower():
            linker_type = "reranker"
        elif "instruct" in model_name.lower():
            linker_type = "instruct"
        else:
            linker_type = "semantic"
    
    logger.info(f"🔗 Starting Entity Linking for domain={domain}")
    logger.info(f"NER input: {ner_output_dir}")
    logger.info(f"EL output: {el_output_dir}")
    logger.info(f"Linker type: {linker_type}")
    logger.info(f"Taxonomy: {taxonomy_path} (source: {taxonomy_source})")
    logger.info(f"Threshold: {threshold}")
    logger.info(f"Context window: {context_window} {'tokens' if context_window > 0 else '(disabled)'}")
    logger.info(f"Max. contexts: {max_contexts}")
    if context_window > 0:
        logger.info(f"Context type: {'sentences' if use_sentence_context else 'token windows'}")

    # Load checkpoint
    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    
    if resume:
        processed = load_json(checkpoint_file, default={})
        logger.info(f"📦 Checkpoint loaded: {len(processed)} files already processed")
    else:
        processed = {}
        logger.info(f"Processing all files (no files processed or resume=False)")
    
    # Load cache
    cache_file = os.path.join(el_output_dir, "cache/linking_cache.json")
    cache = load_json(cache_file, default={})
    logger.info(f"💾 Cache loaded: {len(cache)} entries")

    # Initialize linker based on type
    if linker_type == "reranker":
        from src.reranker_linker import RerankerLinker
        
        logger.info(f"Reranker configuration:")
        logger.info(f"  Embedding model: {model_name}")
        logger.info(f"  LLM model: {reranker_llm}")
        logger.info(f"  Top-k candidates: {reranker_top_k}")
        logger.info(f"  Context for retrieval: {use_context_for_retrieval}")  # NEW
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
            use_context_for_retrieval=use_context_for_retrieval,  # NEW
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
            #log_memory_usage(logger, f"before file {files_processed + 1}")
        
            # Load NER entities
            #logger.info(f"  Loading NER output {ner_file}")
            df_ner = pd.read_json(ner_file, lines=True)
            
            # Load corresponding expanded text
            base_name = os.path.basename(ner_file).replace('.jsonl', '_sections.csv')
            expanded_file = os.path.join(expanded_dir, base_name)
            
            if not os.path.exists(expanded_file):
                logger.warning(f"⚠️  No expanded file for {ner_file}")
                processed[ner_file] = {"status": "no_expanded"}
                continue

            # DEBUG !!!
            #logger.info(f"  Loading expanded file {expanded_file}")
            #log_memory_usage(logger, "before CSV load")
                
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
                logger.info(f"  ✓ Loaded {len(df_expanded)} sections successfully")
                #log_memory_usage(logger, "after CSV load")
                
            except Exception as csv_error:
                logger.error(f"❌ Failed to load CSV: {csv_error}")
                import traceback
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
            
            for _, row in df_ner.iterrows():
                section_id = row['section_id']
                entities = row['entities']
                section_text = text_map.get(section_id, "")
                
                if not section_text:
                    linked_sections.append(row.to_dict())
                    continue
                
                # Link entities with taxonomy source
                # DEBUG !!!
                #logger.info(f"  Linking section {section_id}")
                enriched_entities, cache = linker.link_entities_in_section(
                    section_text, 
                    entities, 
                    cache, 
                    taxonomy_source,
                    filename=os.path.basename(ner_file),
                    section_id=section_id
                )
                               
                # Count statistics
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
            
            processed[ner_file] = {
                "status": "done",
                "total_entities": total_entities,
                "linked_entities": total_linked,
                "linking_rate": f"{100*total_linked/total_entities:.1f}%" if total_entities > 0 else "0%"
            }
            
            logger.info(
                f"✅ {os.path.basename(ner_file)}: "
                f"{total_linked}/{total_entities} entities linked "
                f"({100*total_linked/total_entities:.1f}%)"
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
            import traceback
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
    
    logger.info(f"📊 Summary: {total_files} files, {total_ents} entities, {total_links} linked")
    if total_ents > 0:
        logger.info(f"   Overall linking rate: {100*total_links/total_ents:.1f}%")

def main():
    parser = argparse.ArgumentParser(description="SciLake NER & Entity Linking Pipeline")
    parser.add_argument("--domain", required=True, help="Domain name (ccam, energy, etc.)")
    parser.add_argument("--input", help="Path to input NIF directory (for NER step)")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--step", default="all", help="gaz | ner | el | all")
    
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
    
    # NEW: Reranker-specific arguments
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

    if args.step == "gaz":
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
        print(f"❌ Unknown step: {args.step}. Use: ner | el | all")

if __name__ == "__main__":
    main()
