# src/ner_runner.py
from __future__ import annotations

import re
import os
from typing import List, Dict, Any, Optional
import gc, torch

import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from gliner import GLiNER

# NOTE: configs is a sibling package to src/, so absolute import without "src."
from configs.domain_models import DOMAIN_MODELS

# Optional: only used in __main__ debug CLI
try:
    from src.utils.logger import setup_logger  # noqa
except Exception:
    setup_logger = None  # type: ignore

# ======================================================
# 1) Model Loading and Caching
# ======================================================

PIPELINE_CACHE: Dict[str, Any] = {}
DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("roberta-base")


def get_model(model_info: Dict[str, Any], logger=None):
    """Load and cache GLiNER or RoBERTa models."""
    name = model_info["name"]
    if name in PIPELINE_CACHE:
        if logger:
            logger.debug(f"Using cached model: {name}")
        return PIPELINE_CACHE[name]

    mtype = model_info["type"].lower()
    if logger:
        logger.info(f"Loading {mtype.upper()} model: {name}")

    if mtype == "gliner":
        model = GLiNER.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        if logger:
            logger.info(f"Model {name} loaded to {device.upper()}")
    else:
        model = pipeline(
            "token-classification",
            model=AutoModelForTokenClassification.from_pretrained(name),
            tokenizer=AutoTokenizer.from_pretrained(name),
            aggregation_strategy="simple",
            device=0 if torch.cuda.is_available() else -1,
        )

    PIPELINE_CACHE[name] = model
    return model


# ======================================================
# 2) RoBERTa Post-processing
# ======================================================

def postprocess_roberta(entities, text: Optional[str] = None):
    """
    Merge adjacent RoBERTa subtoken predictions while respecting 'Ġ' boundaries.
    Fixes partial tokens like 'fer' → 'ferrofluid'.
    """
    if not entities:
        return []

    merged, buffer = [], None

    for ent in entities:
        tag = ent.get("entity_group", ent.get("entity", ""))
        raw_word = ent["word"]
        start_new = raw_word.startswith("Ġ")

        # Clean the token text
        word = raw_word.replace("Ġ", "").replace("##", "").strip()
        start, end = ent.get("start"), ent.get("end")

        if buffer is None:
            buffer = ent.copy()
            buffer.update(
                {"word": word, "entity": tag, "start": start, "end": end, "start_new": start_new}
            )
            continue

        same_type = tag == buffer["entity"]
        adjacent = (
            isinstance(start, int)
            and isinstance(buffer.get("end"), int)
            and 0 <= start - buffer["end"] <= 2
        )

        has_space_or_punct = False
        if text and isinstance(start, int) and isinstance(buffer.get("end"), int):
            gap = text[buffer["end"]:start]
            has_space_or_punct = bool(re.match(r"[\s\.,;:]", gap))

        if same_type and adjacent and not start_new and not has_space_or_punct:
            sep = "" if buffer["word"].endswith("-") else ""
            buffer["word"] += sep + word
            buffer["end"] = end
            buffer["score"] = max(buffer["score"], ent["score"])
        else:
            merged.append(buffer)
            buffer = ent.copy()
            buffer.update(
                {"word": word, "entity": tag, "start": start, "end": end, "start_new": start_new}
            )

    if buffer:
        merged.append(buffer)

    # Expand short fragments
    for e in merged:
        w = e["word"].lower()
        if text and len(w) <= 4:
            snippet = text[e["start"]: e["start"] + 30]
            m = re.match(rf"{re.escape(w)}[a-zA-Z\-]{{3,}}", snippet)
            if m:
                e["word"] = m.group(0)
        e["word"] = re.sub(r"\s+", " ", e["word"]).strip()

    return merged


# ======================================================
# 3) Text Chunking
# ======================================================

def chunk_text(text: str, max_tokens: int = 512, stride: int = 50):
    """Split text into overlapping chunks with offset mapping."""
    tokens = DEFAULT_TOKENIZER(
        text, return_offsets_mapping=True, add_special_tokens=False, truncation=False
    )
    offsets = tokens["offset_mapping"]
    input_ids = tokens["input_ids"]

    chunks = []
    start_idx = 0
    while start_idx < len(input_ids):
        end_idx = min(start_idx + max_tokens, len(input_ids))
        chunk_ids = input_ids[start_idx:end_idx]
        chunk_offsets = offsets[start_idx:end_idx]
        if not chunk_offsets:
            break

        char_start, char_end = chunk_offsets[0][0], chunk_offsets[-1][1]
        chunk_text_str = DEFAULT_TOKENIZER.decode(chunk_ids)
        chunks.append({"text": chunk_text_str, "char_start": char_start, "char_end": char_end})

        if end_idx == len(input_ids):
            break
        start_idx += max_tokens - stride

    return chunks


# ======================================================
# 4) Entity Merging
# ======================================================

def merge_adjacent_entities(entities: List[Dict[str, Any]], max_gap: int = 2):
    """Merge nearby entities of the same type and model."""
    if not entities:
        return []

    entities = sorted(entities, key=lambda e: e.get("start", float("inf")))
    merged, buffer = [], None

    for ent in entities:
        etype, model, text = (
            ent.get("entity", "").lower(),
            ent.get("model", ""),
            ent.get("text", "").strip(),
        )
        start, end = ent.get("start"), ent.get("end")

        if buffer is None:
            buffer = ent.copy()
            buffer["entity"] = etype
            continue

        same_type = etype == buffer["entity"]
        same_model = model == buffer["model"]
        adjacent = (
            isinstance(start, int)
            and isinstance(buffer.get("end"), int)
            and 0 <= start - buffer["end"] <= max_gap
        )

        if same_type and same_model and adjacent:
            buffer["text"] += " " + text
            buffer["end"] = end or buffer["end"]
        else:
            merged.append(buffer)
            buffer = ent.copy()
            buffer["entity"] = etype

    if buffer:
        merged.append(buffer)
    return merged


# ======================================================
# 5) Batch Prediction
# ======================================================

def predict_entities_batch(
    texts: List[str],
    offsets: List[int],
    domain: str,
    model_conf: Dict[str, Any],
    section_ids: List[str],
    logger=None,
) -> List[Dict[str, Any]]:
    """Predict entities for a batch of texts, keeping correct section mapping."""
    model = get_model(model_conf, logger=logger)
    model_type = model_conf["type"].lower()
    threshold = float(model_conf.get("threshold", 0.5))
    domain_conf = DOMAIN_MODELS[domain]
    results: List[Dict[str, Any]] = []

    labels_dict: Dict[str, List[str]] = domain_conf.get("labels", {})

    if model_type == "gliner":
        # Use domain label list if provided (your config uses per-domain)
        labels = labels_dict.get(domain, []) or labels_dict.get("gliner", [])

        # Prefer the new API; it returns a list aligned with input texts
        try:
            batch_entities = model.run(texts, labels=labels, threshold=threshold)
        except Exception:
            # Fallback for older versions (deprecated)
            if logger:
                logger.warning("GLiNER.run failed; falling back to batch_predict_entities (deprecated).")
            batch_entities = model.batch_predict_entities(
                texts, labels, flat_ner=True, threshold=threshold
            )

        if logger:
            logger.debug(
                f"🧩 DEBUG[GLINER] texts={len(texts)} offsets={len(offsets)} "
                f"sections={len(section_ids)} preds={len(batch_entities) if batch_entities else 0}"
            )

        n = min(len(batch_entities or []), len(offsets), len(section_ids))
        if (len(batch_entities or []) != len(texts)) and logger:
            logger.warning(
                f"⚠️ GLiNER mismatch: texts={len(texts)} preds={len(batch_entities or [])} "
                f"(model={model_conf['name']})"
            )

        for i in range(n):
            ents = batch_entities[i] or []
            offset = offsets[i]
            sid = section_ids[i]
            for e in ents:
                results.append(
                    {
                        "entity": (e.get("label") or "").lower(),
                        "text": e.get("text", ""),
                        "score": float(e.get("score", 1.0)),
                        "start": (e.get("start") or 0) + offset,
                        "end": (e.get("end") or 0) + offset,
                        "model": model_conf["name"],
                        "domain": domain,
                        "section_id": sid,
                    }
                )

    elif model_type == "roberta":
        preds_batch = model(texts, batch_size=len(texts))
        # Defensive: ensure list length and structure
        preds_batch = preds_batch if isinstance(preds_batch, list) else []
        preds_batch = [
            postprocess_roberta(preds, text=texts[i]) if i < len(texts) else []
            for i, preds in enumerate(preds_batch)
        ]

        if logger:
            logger.debug(
                f"🧩 DEBUG[ROBERTA] texts={len(texts)} offsets={len(offsets)} "
                f"sections={len(section_ids)} preds={len(preds_batch)}"
            )

        n = min(len(preds_batch), len(offsets), len(section_ids))
        if (len(preds_batch) != len(texts)) and logger:
            logger.warning(
                f"⚠️ RoBERTa mismatch: texts={len(texts)} preds={len(preds_batch)} "
                f"(model={model_conf['name']})"
            )

        for i in range(n):
            preds = preds_batch[i] or []
            offset = offsets[i]
            sid = section_ids[i]
            for p in preds:
                if float(p.get("score", 0.0)) >= threshold:
                    raw = p.get("word", "")
                    clean_text = (
                        raw.replace(" ##", "").replace("Ġ", "").replace(" - ", "-").strip()
                    )
                    results.append(
                        {
                            "entity": p.get("entity_group", p.get("entity")),
                            "text": clean_text,
                            "score": float(p.get("score", 0.0)),
                            "start": (p.get("start") or 0) + offset,
                            "end": (p.get("end") or 0) + offset,
                            "model": model_conf["name"],
                            "domain": domain,
                            "section_id": sid,
                        }
                    )

    return results


# ======================================================
# 6) Full Section Prediction Pipeline
# ======================================================

def predict_sections_multimodel(
    df: pd.DataFrame,
    domain: str = "energy",
    text_col: str = "section_content_expanded",
    id_col: str = "section_id",
    stride: int = 25,
    batch_size: int = 8,
    logger=None,
) -> pd.DataFrame:
    """Predict entities for all sections using GLiNER + RoBERTa models."""
    domain_conf = DOMAIN_MODELS.get(domain)
    if not domain_conf:
        raise ValueError(f"Unknown domain '{domain}'")

    models = domain_conf.get("models", [])
    results_all: List[Dict[str, Any]] = []
    chunk_config = {"gliner": 385, "roberta": 512}

    for model_conf in models:
        model_type = model_conf["type"].lower()
        model_name = model_conf["name"]
        max_tokens = chunk_config.get(model_type, 512)
        if logger:
            logger.info(f"🔹 Running {model_type.upper()} model: {model_name}")
        else:
            tqdm.write(f"\n🔹 Running {model_type.upper()} model: {model_name}")

        # --- Chunk sections ---
        rows = []
        it = tqdm(df.iterrows(), total=len(df), desc=f"Chunking for {model_name}")
        for _, row in it:
            section_id = row[id_col]
            text = row[text_col]
            if not isinstance(text, str) or not text.strip():
                continue
            chunks = chunk_text(text, max_tokens=max_tokens, stride=stride)
            for ch in chunks:
                rows.append({"section_id": section_id, "text": ch["text"], "offset": ch["char_start"]})

        if not rows:
            continue
        df_chunks = pd.DataFrame(rows)

        # --- Predict in batches ---
        all_ents: List[Dict[str, Any]] = []
        for i in tqdm(range(0, len(df_chunks), batch_size), desc=f"Predicting [{model_type}]"):
            batch = df_chunks.iloc[i : i + batch_size]
            texts = batch["text"].tolist()
            offsets = batch["offset"].tolist()
            section_ids = batch["section_id"].tolist()

            ents = predict_entities_batch(
                texts, offsets, domain, model_conf, section_ids, logger=logger
            )
            all_ents.extend(ents)

            torch.cuda.empty_cache()
            gc.collect()

        df_pred = pd.DataFrame(all_ents)
        if df_pred.empty:
            continue

        # Merge adjacent entities per section for this model
        for sid, group in df_pred.groupby("section_id"):
            merged = merge_adjacent_entities(group.to_dict("records"))
            results_all.append({"section_id": sid, "entities": merged, "model": model_name})

    if not results_all:
        return pd.DataFrame(columns=["section_id", "entities"])

    df_all = pd.DataFrame(results_all)

    # --- Combine model outputs per section ---
    if logger:
        logger.info("🔹 Combining GLiNER + RoBERTa entities per section...")
    else:
        tqdm.write("\n🔹 Combining GLiNER + RoBERTa entities per section...")

    merged_rows = []
    for section_id in df_all["section_id"].unique():
        group = df_all[df_all["section_id"] == section_id]
        combined_entities: List[Dict[str, Any]] = []
        for ents in group["entities"]:
            if isinstance(ents, list):
                combined_entities.extend(ents)
        combined_entities = [e for e in combined_entities if isinstance(e, dict)]
        combined_entities = sorted(combined_entities, key=lambda x: x.get("start", float("inf")))
        merged = merge_adjacent_entities(combined_entities)
        merged_rows.append({"section_id": section_id, "entities": merged})

    df_final = pd.DataFrame(merged_rows)
    if logger:
        logger.info(f"✅ Completed final merge — {len(df_final)} unified sections.")
    else:
        tqdm.write(f"\n✅ Completed final merge — {len(df_final)} unified sections.")
    return df_final


# ======================================================
# 7) CLI (optional debug)
# ======================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NER models on a CSV of sections.")
    parser.add_argument("--input", required=True, help="CSV file with expanded sections")
    parser.add_argument("--domain", required=True, help="Domain name (e.g., energy, ccam)")
    parser.add_argument("--text_col", default="section_content_expanded")
    parser.add_argument("--id_col", default="section_id")
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    log = None
    if setup_logger:
        os.makedirs("logs", exist_ok=True)
        log = setup_logger("logs", "ner_runner_cli")

    df_in = pd.read_csv(args.input)
    df_out = predict_sections_multimodel(
        df_in,
        domain=args.domain,
        text_col=args.text_col,
        id_col=args.id_col,
        stride=args.stride,
        batch_size=args.batch_size,
        logger=log,
    )
    out_path = f"outputs/{args.domain}_ner_results.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_json(out_path, orient="records", lines=True, force_ascii=False)
    if log:
        log.info(f"✅ Saved NER output to {out_path}")
    else:
        print(f"✅ Saved NER output to {out_path}")
