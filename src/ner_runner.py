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
            aggregation_strategy="none",
            device=0 if torch.cuda.is_available() else -1,
        )

    PIPELINE_CACHE[name] = model
    return model

# ======================================================
# 2) AIONER post-processing
# ======================================================

def process_aioner_output(pipeline_output, pipeline, text, entity_type='ALL'):
    entity_tag = '<{}>'.format(entity_type)
    closing_tag = '</{}>'.format(entity_type)
    offset = len(entity_tag)

    index_offset = min([i for i, x in enumerate(pipeline.tokenizer.tokenize(entity_tag + text + closing_tag)) if x=='>'])

    results = []
    for entity in pipeline_output:
        if entity['entity'].startswith('O') or entity['word'].startswith('##'):
            continue

        entity['start'] -= offset
        entity['end'] -= offset
        entity['index'] -= index_offset

        results.append(entity)

    if results:
        for entity in results:
            entity['word'] = complete_subwords(entity['word'], entity['index']-1, pipeline.tokenizer.tokenize(text))
            entity['end'] = entity['start'] + len(entity['word'])
        return pipeline.group_entities(results)
    return []

def complete_subwords(word, word_index, token_list):
    output_word = word
    idx = word_index
    while idx<len(token_list):
        if token_list[idx].startswith('##'):
            output_word+=token_list[idx][2:]
            idx+=1
        else:
            break
    
    return output_word


# ======================================================
# 3) RoBERTa Post-processing
# ======================================================

def process_roberta_output(pipeline_output, text):
    
    words = reconstruct_words_with_completion(pipeline_output, text)
    
    entities = merge_entities(words, text)

    # Filter out invalid entities
    filtered_entities = []
    for ent in entities:
        word = ent.get('word', '').strip()
        
        # Skip empty
        if not word:
            continue
        
        # Skip pure punctuation (e.g., ".", ",")
        if not any(c.isalnum() for c in word):
            continue
        
        # Skip pure numbers including decimals (e.g., "45.8", "3.2", "0.82")
        if re.match(r'^-?\d+([.,]\d+)?%?$', word):
            continue
        
        # Skip single characters (e.g., "a", "s")
        if len(word) <= 1:
            continue
        
        # Skip common stopwords that shouldn't be entities
        stopwords = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                     'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'under', 'again', 'further', 'then', 'once'}
        if word.lower() in stopwords:
            continue
            
        filtered_entities.append(ent)
    
    entities = filtered_entities

    # Parentheses balancing
    for ent in entities:
        start, end, word = ent['start'], ent['end'], ent['word']
        if word.count("(") > word.count(")") and end < len(text) and text[end] == ")":
            end += 1
            word += ")"
        elif word.count(")") > word.count("(") and start > 0 and text[start-1] == "(":
            start -= 1
            word = "(" + word
        
        ent["start"], ent["end"], ent["word"] = start, end, word
        
    return entities

def reconstruct_words_with_completion(pipeline_output, text):
    """
    Reconstruct complete words, detecting and fixing incomplete words
    """
    words = []
    current_word = None
    
    for i, token_data in enumerate(pipeline_output):
        # Skip special tokens
        if token_data['word'] in ['<s>', '</s>', '<pad>']:
            continue
        
        # Check if new word (has Ġ prefix OR not consecutive index)
        is_new_word = token_data['word'].startswith('Ġ')
        
        # ALSO check index continuity
        if not is_new_word and current_word:
            if 'last_index' in current_word:
                # Multi-token word - check against last token's index
                if token_data['index'] != current_word['last_index'] + 1:
                    is_new_word = True
            else:
                # Single token word so far - check against its index
                if token_data['index'] != current_word['index'] + 1:
                    is_new_word = True
        
        if is_new_word:
            if current_word:
                # Check if word is complete before saving
                current_word2 = current_word.copy()
                current_word = complete_word_if_needed(current_word, text)
                words.append(current_word)
            
            current_word = {
                'word': token_data['word'].lstrip('Ġ'),
                'entity': token_data['entity'],
                'score': token_data['score'],
                'start': token_data['start'],
                'end': token_data['end'],
                'index': token_data['index']
            }
        else:
            # Continue current word (subword without Ġ AND consecutive index)
            if current_word:
                current_word['word'] += token_data['word']
                current_word['end'] = token_data['end']
                current_word['score'] = (current_word['score'] + token_data['score']) / 2
                current_word['last_index'] = token_data['index']
            else:
                # First token
                current_word = {
                    'word': token_data['word'],
                    'entity': token_data['entity'],
                    'score': token_data['score'],
                    'start': token_data['start'],
                    'end': token_data['end'],
                    'index': token_data['index']
                }
    
    if current_word:
        current_word2 = current_word.copy()
        current_word = complete_word_if_needed(current_word, text)
        words.append(current_word)
    
    return words

def complete_word_if_needed(word_data, text): 
    """
    Check if word is incomplete and extend to full word boundary (both directions)
    Keeps hyphens/underscores but removes parentheses and other punctuation
    """
    start = word_data['start']
    end = word_data['end']
    
    # Check backward - are we starting mid-word?
    if start > 0:
        # Look backwards for word boundary
        while start > 0 and (text[start-1].isalnum()):
            start -= 1
    
    # Check forward - are we ending mid-word?
    if end < len(text):
        # Look forwards for word boundary
        while end < len(text) and (text[end].isalnum()):
            end += 1
    
    # Also strip any leading/trailing parentheses that might already be in the span
    while start < end and text[start] in '()[]{}':
        start += 1
    while end > start and text[end-1] in '()[]{}':
        end -= 1
    
    # Update word
    word_data['word'] = text[start:end]
    word_data['start'] = start
    word_data['end'] = end
    
    return word_data

def merge_entities(words, text):
    """Merge words into multi-word entities based on BIO tags"""
    entities = []
    current_entity = None

    for word in words:
        if word['entity'].startswith('B-'):
            entity_type = word['entity'][2:]
            
            if current_entity and current_entity['entity'] == entity_type:
                index_gap = word.get('index', -999) - current_entity.get('last_index', current_entity.get('index', -999))
                
                # Check if consecutive OR has hyphen/underscore connector
                if index_gap == 1:
                    # Directly consecutive - merge
                    gap = text[current_entity['end']:word['start']]
                    current_entity['word'] = current_entity['word'] + gap + word['word']
                    current_entity['end'] = word['end']
                    current_entity['score'] = (current_entity['score'] + word['score']) / 2
                    current_entity['last_index'] = word.get('last_index', word.get('index', -999))
                elif index_gap == 2:
                    # One token between - check if it's a connector
                    gap = text[current_entity['end']:word['start']]
                    if gap.strip() in ['-', '_', '–']:  # hyphen, underscore, en-dash
                        # Merge including the connector
                        current_entity['word'] = current_entity['word'] + gap + word['word']
                        current_entity['end'] = word['end']
                        current_entity['score'] = (current_entity['score'] + word['score']) / 2
                        current_entity['last_index'] = word.get('last_index', word.get('index', -999))
                    else:
                        # Something else between - don't merge
                        entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                        current_entity = {
                            'entity': entity_type,
                            'word': word['word'],
                            'score': word['score'],
                            'start': word['start'],
                            'end': word['end'],
                            'last_index': word.get('last_index', word.get('index', -999))
                        }
                else:
                    # Too far apart - save old, start new
                    entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                    current_entity = {
                        'entity': entity_type,
                        'word': word['word'],
                        'score': word['score'],
                        'start': word['start'],
                        'end': word['end'],
                        'last_index': word.get('last_index', word.get('index', -999))
                    }
            else:
                if current_entity:
                    entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                current_entity = {
                    'entity': entity_type,
                    'word': word['word'],
                    'score': word['score'],
                    'start': word['start'],
                    'end': word['end'],
                    'last_index': word.get('last_index', word.get('index', -999))
                }
                
        elif word['entity'].startswith('I-'):
            # Same logic for I- tags
            if current_entity and word['entity'][2:] == current_entity['entity']:
                index_gap = word.get('index', -999) - current_entity.get('last_index', current_entity.get('index', -999))
                
                
                if index_gap == 1:
                    # Consecutive
                    gap = text[current_entity['end']:word['start']]
                    current_entity['word'] = current_entity['word'] + gap + word['word']
                    current_entity['end'] = word['end']
                    current_entity['score'] = (current_entity['score'] + word['score']) / 2
                    current_entity['last_index'] = word.get('last_index', word.get('index', -999))
                elif index_gap == 2:
                    # Check for connector
                    gap = text[current_entity['end']:word['start']]
                    if gap.strip() in ['-', '_', '–']:
                        current_entity['word'] = current_entity['word'] + gap + word['word']
                        current_entity['end'] = word['end']
                        current_entity['score'] = (current_entity['score'] + word['score']) / 2
                        current_entity['last_index'] = word.get('last_index', word.get('index', -999))
                    else:
                        if current_entity:
                            entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                        current_entity = {
                            'entity': word['entity'][2:],
                            'word': word['word'],
                            'score': word['score'],
                            'start': word['start'],
                            'end': word['end'],
                            'last_index': word.get('last_index', word.get('index', -999))
                        }
                else:
                    # Too far apart
                    if current_entity:
                        entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                    current_entity = {
                        'entity': word['entity'][2:],
                        'word': word['word'],
                        'score': word['score'],
                        'start': word['start'],
                        'end': word['end'],
                        'last_index': word.get('last_index', word.get('index', -999))
                    }
            else:
                if current_entity:
                    entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
                current_entity = {
                    'entity': word['entity'][2:],
                    'word': word['word'],
                    'score': word['score'],
                    'start': word['start'],
                    'end': word['end'],
                    'last_index': word.get('last_index', word.get('index', -999))
                }
    
    if current_entity:
        entities.append({k: v for k, v in current_entity.items() if k != 'last_index'})
    
    return entities


# ======================================================
# 4) Text Chunking
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
# 5) Entity Merging
# ======================================================

def merge_adjacent_entities(entities: List[Dict[str, Any]], max_gap: int = 1):
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
# 6) Batch Prediction
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
            
                entity_start = (e.get("start") or 0)
                entity_end = (e.get("end") or 0)
                final_start = entity_start + offset
                final_end = entity_end + offset
                
                if logger:
                    logger.debug(
                        f"  Entity '{e.get('text')}': chunk_pos=({entity_start},{entity_end}) "
                        f"+ offset={offset} → final=({final_start},{final_end})"
                    )
            
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
            process_roberta_output(preds, text=texts[i]) if i < len(texts) else []
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
                    
                    entity_start = p.get("start") or 0
                    entity_end = p.get("end") or 0
                    final_start = entity_start + offset
                    final_end = entity_end + offset
                    
                    if logger:
                        logger.debug(
                            f"  Entity '{clean_text}': chunk_pos=({entity_start},{entity_end}) "
                            f"+ offset={offset} → final=({final_start},{final_end})"
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

    elif model_type == "aioner":
        entity_type = 'ALL'
        entity_tag = '<{}>'.format(entity_type)
        closing_tag = '</{}>'.format(entity_type)

        preds_batch = model([entity_tag+text+closing_tag for text in texts], batch_size=len(texts))
        # Defensive: ensure list length and structure
        preds_batch = preds_batch if isinstance(preds_batch, list) else []
        preds_batch = [
            process_aioner_output(preds, pipeline=model, text=texts[i]) if i < len(texts) else []
            for i, preds in enumerate(preds_batch)
        ]

        if logger:
            logger.debug(
                f"🧩 DEBUG[AIONER] texts={len(texts)} offsets={len(offsets)} "
                f"sections={len(section_ids)} preds={len(preds_batch)}"
            )

        n = min(len(preds_batch), len(offsets), len(section_ids))
        if (len(preds_batch) != len(texts)) and logger:
            logger.warning(
                f"⚠️ AIONER mismatch: texts={len(texts)} preds={len(preds_batch)} "
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
                    
                    entity_start = p.get("start") or 0
                    entity_end = p.get("end") or 0
                    final_start = entity_start + offset
                    final_end = entity_end + offset
                    
                    if logger:
                        logger.debug(
                            f"  Entity '{clean_text}': chunk_pos=({entity_start},{entity_end}) "
                            f"+ offset={offset} → final=({final_start},{final_end})"
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
# 7) Full Section Prediction Pipeline
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
# 8) CLI (optional debug)
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
