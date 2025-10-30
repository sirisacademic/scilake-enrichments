import argparse
import os
import pandas as pd
import traceback
from tqdm import tqdm

from src.utils.logger import setup_logger
from src.utils.io_utils import load_json, save_json, append_jsonl
from src.nif_reader import parse_nif_file, apply_acronym_expansion
from src.ner_runner import predict_sections_multimodel
from configs.domain_models import DOMAIN_MODELS


def run_ner(domain, input_dir, output_dir, resume=True, file_batch_size=100):
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
    logger = setup_logger(log_dir, name=f"{domain}_ner")

    checkpoint_file = os.path.join(checkpoint_dir, "processed.json")
    processed = load_json(checkpoint_file, default={})

    logger.info(f"✅ Starting NER for domain={domain}")
    logger.info(f"Input dir: {input_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Resuming from checkpoint: {len(processed)} files already done")

    # --- find input files ---
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(input_dir)
        for f in files if f.endswith(".ttl")
    ]
    remaining_files = [f for f in all_files if f not in processed]
    total = len(remaining_files)
    logger.info(f"Found {len(all_files)} files ({total} pending)")

    # --- batch iteration ---
    for start in range(0, total, file_batch_size):
        batch_files = remaining_files[start:start + file_batch_size]
        logger.info(f"\n🧩 Processing batch {start // file_batch_size + 1} "
                    f"({len(batch_files)} files)...")

        all_sections = []
        paper_map = {}  # section_id → paper_path

        # 1️⃣ Parse all papers in the batch
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

        # 2️⃣ Run NER once for all sections in batch
        try:
            df_entities = predict_sections_multimodel(
                df_all,
                domain=domain,
                text_col="section_content_expanded",
                id_col="section_id",
                stride=10,
                batch_size=8,
                logger=logger,
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"🔥 NER failed on batch: {e}")
            logger.debug(tb)
            continue

        # 3️⃣ Group entities per paper and save
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

def main():
    parser = argparse.ArgumentParser(description="SciLake NER & Enrichment Pipeline")
    parser.add_argument("--domain", required=True, help="Domain name (ccam, energy, etc.)")
    parser.add_argument("--input", required=True, help="Path to input NIF directory")
    parser.add_argument("--output", required=True, help="Path to output directory")
    parser.add_argument("--step", default="ner", help="ner | link | all")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=1000, help="Files per batch")

    args = parser.parse_args()

    if args.step == "ner":
        run_ner(
            domain=args.domain,
            input_dir=args.input,
            output_dir=os.path.join(args.output, "ner"),
            resume=args.resume,
            file_batch_size=args.batch_size,
        )
    else:
        print("Only NER implemented so far — Linking coming soon.")


if __name__ == "__main__":
    main()