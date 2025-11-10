import pandas as pd
from transformers import pipeline, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
import torch
import gc
from tqdm.auto import tqdm

# Ensure NLTK sentence tokenizer data
nltk.download("punkt", quiet=True)

class GeordieNER:
    def __init__(self, device=-1):
        self.model = AutoModelForTokenClassification.from_pretrained("SIRIS-Lab/geordie-ner")
        self.tokenizer = AutoTokenizer.from_pretrained("SIRIS-Lab/geordie-ner")
        self.tokenizer.model_max_length = 512
        self.ner_pipeline = pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            task="ner",
            aggregation_strategy="simple",
            device=device,
        )

    def extract_entities_from_corpus(self, texts, batch_size=8):
        return self.ner_pipeline(texts, batch_size=batch_size)

class RoleClassifier:
    def __init__(self, device=-1):
        self.model = AutoModelForSequenceClassification.from_pretrained("SIRIS-Lab/geordie-role")
        self.tokenizer = AutoTokenizer.from_pretrained("SIRIS-Lab/geordie-role")
        self.tokenizer.model_max_length = 512
        self.role_pipeline = pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-classification", device=device
        )

    def classify_role(self, entities_in_sentence):
        results = []
        for item in entities_in_sentence:
            context = item["context"]
            role_type = self.role_pipeline(context)
            item["role"] = role_type
            results.append(item)
        return results


# Helper to get local sentence context
def get_context_of_the_mention(text, entities):
    """
    For each entity, find the sentence containing it and use that as context.
    """
    sentences = sent_tokenize(text)
    results = []
    for ent in entities:
        entity_text = ent["word"]
        start = ent.get("start")
        end = ent.get("end")
        for sent in sentences:
            if entity_text in sent:
                sentence_marked = sent.replace(entity_text, f"[START_ENT] {entity_text} [END_ENT]")
                results.append(
                    {
                        "entity": entity_text,
                        "entity_normalised": ent.get("entity_group", entity_text),
                        "context": sent,
                        "start": start,
                        "end": end,
                    }
                )
                break
    return results


# Use the same tokenizer as your Geordie NER model
DEFAULT_TOKENIZER = AutoTokenizer.from_pretrained("SIRIS-Lab/geordie-ner")


def chunk_text(text: str, max_tokens: int = 512, stride: int = 50):
    """
    Split long text into overlapping chunks based on tokenizer length.
    Returns a list of dicts with chunk text and character offsets.
    """
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


def merge_chunk_entities(entities_per_chunk):
    """
    Merge entities from overlapping chunks by deduplicating based on text span.
    """
    merged = []
    seen = set()
    for ent in sorted(entities_per_chunk, key=lambda x: x.get("start", 0)):
        key = (ent.get("entity_group"), ent.get("start"), ent.get("end"))
        if key not in seen:
            seen.add(key)
            merged.append(ent)
    return merged


def run_geotagging_batch(
    df_sections: pd.DataFrame,
    logger=None,
    text_col: str = "section_content_expanded",
    id_col: str = "section_id",
    batch_size: int = 8,
    device: str = "cpu",
    stride: int = 50,
):
    """
    Run Geotagging pipeline (GeoNER + Role Classification) directly, with text chunking.
    """

    device_index = 0 if device == "cuda" else -1
    ner_model = GeordieNER(device=device_index)
    role_model = RoleClassifier(device=device_index)

    logger and logger.info(f"🌍 Running Geotagging on {len(df_sections)} sections (device={device})")

    texts = df_sections[text_col].tolist()
    section_ids = df_sections[id_col].tolist()

    all_entities = []

    # ======================================================
    # Step 1️⃣ Chunk each section and run GeoNER
    # ======================================================
    logger and logger.info(f"🧠 Running GeoNER model on {len(texts)} sections with chunking...")

    rows = []
    for sid, text in tqdm(zip(section_ids, texts), total=len(texts), desc="Chunking texts"):
        if not isinstance(text, str) or not text.strip():
            continue
        chunks = chunk_text(text, max_tokens=512, stride=stride)
        for ch in chunks:
            rows.append({"section_id": sid, "text": ch["text"], "offset": ch["char_start"]})

    if not rows:
        logger and logger.warning("⚠️ No text to process after chunking.")
        return pd.DataFrame(columns=["section_id", "entities"])

    df_chunks = pd.DataFrame(rows)

    entities_per_section = {}
    for i in tqdm(range(0, len(df_chunks), batch_size), desc="Running GeoNER"):
        batch = df_chunks.iloc[i : i + batch_size]
        texts_batch = batch["text"].tolist()
        offsets = batch["offset"].tolist()
        sids = batch["section_id"].tolist()

        preds = ner_model.extract_entities_from_corpus(texts_batch, batch_size=batch_size)
        preds = preds if isinstance(preds, list) else []

        n = min(len(preds), len(texts_batch))
        for j in range(n):
            sid = sids[j]
            offset = offsets[j]
            ents = preds[j] or []
            adjusted = []
            for e in ents:
                start = e.get("start", 0) + offset
                end = e.get("end", 0) + offset
                adjusted.append(
                    {
                        "entity_group": e.get("entity_group", e.get("entity")),
                        "word": e.get("word", ""),
                        "start": start,
                        "end": end,
                        "score": e.get("score", 0.0),
                    }
                )
            entities_per_section.setdefault(sid, []).extend(adjusted)

        torch.cuda.empty_cache()
        gc.collect()

    # Merge overlapping entities from multiple chunks
    for sid, ents in entities_per_section.items():
        entities_per_section[sid] = merge_chunk_entities(ents)

    # ======================================================
    # Step 2️⃣ Build contexts for role classification
    # ======================================================
    contexts_for_role = []
    for sid, text in zip(section_ids, texts):
        ents = entities_per_section.get(sid, [])
        if not ents:
            continue
        entities_in_context = get_context_of_the_mention(text, ents)
        for e in entities_in_context:
            e["section_id"] = sid
        contexts_for_role.extend(entities_in_context)

    if not contexts_for_role:
        logger and logger.info("⚪ No geographical entities detected in this batch.")
        return pd.DataFrame({"section_id": section_ids, "entities": [[] for _ in section_ids]})

    # ======================================================
    # Step 3️⃣ Role classification
    # ======================================================
    logger and logger.info(f"🎯 Running role classification for {len(contexts_for_role)} mentions...")
    classified_contexts = role_model.classify_role(contexts_for_role)

    # ======================================================
    # Step 4️⃣ Group back per section
    # ======================================================
    section_entities = {}
    for item in classified_contexts:
        sid = item["section_id"]
        role_label = None
        role_score = None
        if isinstance(item["role"], list) and len(item["role"]) > 0:
            role_label = item["role"][0]["label"]
            role_score = item["role"][0]["score"]

        section_entities.setdefault(sid, []).append(
            {
                "entity": item.get("entity_normalised", item.get("entity")),
                "text": item["entity"],
                "role": role_label,
                "score": role_score,
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )

    df_out = pd.DataFrame(
        {"section_id": section_ids, "entities": [section_entities.get(sid, []) for sid in section_ids]}
    )

    logger and logger.info(f"✅ Geotagging complete — {len(df_out)} sections processed.")
    return df_out