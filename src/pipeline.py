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


def run_ner(domain, input_dir, output_dir, resume=True, file_batch_size=1000):
    """
    Run NER enrichment with batching and checkpoint.
    Processes NIF files in small batches (e.g., 1k files) to avoid memory blowup.
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

        for path in tqdm(batch_files, desc=f"Batch {start // file_batch_size + 1}"):
            if path in processed:
                continue

            try:
                logger.info(f"🔍 Reading file: {path}")
                # --- parse single NIF file ---
                records = parse_nif_file(path, logger=logger)

                if not records:
                    logger.warning(f"⚠️ No sections found in {path}")
                    processed[path] = {"status": "empty"}
                    continue

                df_sections = pd.DataFrame(records)
                logger.debug(f"📊 Parsed sections DataFrame shape: {df_sections.shape}")

                df_expanded = apply_acronym_expansion(df_sections, logger=logger)
                logger.debug(f"📖 Expanded acronym DataFrame shape: {df_expanded.shape}")

                # --- run NER ---
                logger.info(f"🧠 Running NER on {len(df_expanded)} sections from {os.path.basename(path)}")
                df_entities = predict_sections_multimodel(
                    df_expanded,
                    domain=domain,
                    text_col="section_content_expanded",
                    id_col="section_id",
                    stride=10,
                    batch_size=8,
                    logger=logger,
                )

                print(df_entities)

                logger.debug(f"🏷️ NER output shape: {df_entities.shape}")

                # --- write results ---
                base_name = os.path.basename(path).replace(".ttl", ".jsonl")
                out_path = os.path.join(output_dir, base_name)
                for _, row in df_entities.iterrows():
                    append_jsonl(row.to_dict(), out_path)

                processed[path] = {"status": "done"}
                logger.info(f"✔️  Finished {path}")

            except Exception as e:
                # 🔥 Full debug dump on error
                tb_str = traceback.format_exc()
                logger.error(f"⚠️ Error on {path}: {e}")
                logger.debug(f"🔍 Full traceback:\n{tb_str}")

                # Add optional extra debugging info if partial data exists
                try:
                    logger.debug(f"Partial DF shapes: sections={df_sections.shape if 'df_sections' in locals() else 'n/a'}, "
                                 f"expanded={df_expanded.shape if 'df_expanded' in locals() else 'n/a'}")
                except Exception:
                    pass

                processed[path] = {"status": f"error: {e}"}

        # save after every batch
        save_json(processed, checkpoint_file)
        logger.info(f"✅ Batch complete — total processed so far: {len(processed)} files.")

    save_json(processed, checkpoint_file)
    logger.info("🎉 All files processed successfully.")


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