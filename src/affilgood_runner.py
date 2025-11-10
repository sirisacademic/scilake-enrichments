# src/affilgood_runner.py
import os
import json
from tqdm import tqdm
import pandas as pd
from src.utils.io_utils import append_jsonl

import numpy as np

def make_json_safe(obj):
    """Recursively convert numpy and other non-serializable types to safe Python equivalents."""
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, (np.generic,)):  # catches np.float32, np.int64, etc.
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def run_affilgood_batch(df_batch, affilgood, logger=None, batch_size=500, output_path=None):
    """
    Run AffilGood enrichment on a batch of affiliations.
    - df_batch: pandas DataFrame with columns ['affiliation_id', 'raw_affiliation']
    - affilgood: initialized AffilGood object
    - output_path: optional JSONL file to append results to
    """
    all_results = []

    logger and logger.info(f"🧠 Running AffilGood on {len(df_batch)} affiliations...")

    # Split into mini-batches to avoid memory spikes
    for start in range(0, len(df_batch), batch_size):
        sub_df = df_batch.iloc[start:start + batch_size]
        texts = sub_df["raw_affiliation"].tolist()
        ids = sub_df["affiliation_id"].tolist()

        # Run AffilGood
        # try:
        #     outputs = affilgood.process(texts)
        # except Exception as e:
        #     logger and logger.error(f"🔥 AffilGood failed on batch {start // batch_size + 1}: {e}")
        #     continue

        # Run AffilGood
        try:
            outputs = affilgood.process(texts)
        except Exception as e:
            logger and logger.error(f"🔥 AffilGood failed on batch {start // batch_size + 1}: {e}")
            continue

        for aff_id, raw, out in zip(ids, texts, outputs):
            out_safe = make_json_safe(out)   # <-- convert numpy types
            result = {
                "affiliation_id": aff_id,
                "raw_affiliation": raw,
                "affilgood_result": out_safe,
            }
            all_results.append(result)

            if output_path:
                append_jsonl(result, output_path)

        logger and logger.info(f"✔️ Processed batch {start // batch_size + 1}: {len(sub_df)} affiliations")

    logger and logger.info(f"✅ AffilGood processing complete — total {len(all_results)} processed.")
    return all_results