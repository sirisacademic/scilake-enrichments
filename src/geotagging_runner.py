import pandas as pd
from transformers import pipeline, AutoModelForTokenClassification, AutoModelForSequenceClassification, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize

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
        for sent in sentences:
            if entity_text in sent:
                results.append(
                    {
                        "entity": entity_text,
                        "entity_normalised": ent.get("entity_group", entity_text),
                        "context": sent,
                    }
                )
                break
    return results


def run_geotagging_batch(
    df_sections: pd.DataFrame,
    logger=None,
    text_col: str = "section_content_expanded",
    id_col: str = "section_id",
    batch_size: int = 8,
    device: str = "cpu",
):
    """
    Run Geotagging pipeline (GeoNER + Role Classification) directly, without Geordie wrapper.
    """
    device_index = 0 if device == "cuda" else -1
    ner_model = GeordieNER(device=device_index)
    role_model = RoleClassifier(device=device_index)

    logger and logger.info(f"🌍 Running Geotagging on {len(df_sections)} sections (device={device})")

    texts = df_sections[text_col].tolist()
    section_ids = df_sections[id_col].tolist()

    # --- Step 1: NER ---
    logger and logger.info(f"🧠 Running GeoNER model on {len(texts)} texts...")
    geo_entities_all = ner_model.extract_entities_from_corpus(texts, batch_size=batch_size)

    if len(geo_entities_all) != len(section_ids):
        logger and logger.warning(
            f"⚠️ Mismatch: {len(geo_entities_all)} predictions vs {len(section_ids)} sections."
        )

    n = min(len(geo_entities_all), len(section_ids))
    enriched_entities = []
    contexts_for_role = []

    # --- Step 2: Build contexts ---
    for i in range(n):
        sid = section_ids[i]
        text = texts[i]
        ents = geo_entities_all[i] or []
        if not ents:
            enriched_entities.append({"section_id": sid, "entities": []})
            continue

        entities_in_context = get_context_of_the_mention(text, ents)
        for e in entities_in_context:
            e["section_id"] = sid
        contexts_for_role.extend(entities_in_context)

    if not contexts_for_role:
        logger and logger.info("⚪ No geographical entities detected in this batch.")
        return pd.DataFrame({"section_id": section_ids, "entities": [[] for _ in section_ids]})

    # --- Step 3: Role classification ---
    logger and logger.info(f"🎯 Running role classification for {len(contexts_for_role)} mentions...")
    classified_contexts = role_model.classify_role(contexts_for_role)

    # --- Step 4: Group by section ---
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
                "entity": item["entity_normalised"],
                "text": item["entity"],
                "role": role_label,
                "score": role_score,
            }
        )

    df_out = pd.DataFrame(
        {"section_id": section_ids, "entities": [section_entities.get(sid, []) for sid in section_ids]}
    )

    logger and logger.info(f"✅ Geotagging complete — {len(df_out)} sections processed.")
    return df_out
