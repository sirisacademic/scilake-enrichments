"""
Geographical entity linking using OSM Nominatim.
Links NER-detected geographical entities (e.g. countries, cities, regions)
to structured OSM metadata with caching and normalization.

Reads geotagging JSONL outputs and enriches them with coordinates,
country, and Wikidata references.
"""

import os
import re
import time
import json
import logging
import pandas as pd
from tqdm.auto import tqdm
from typing import Dict, List, Optional

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from collections import OrderedDict
from src.utils.demonyms_and_adjectives import pattern, base_transformed_dict

# -------------------------------------------------------------------
# 🌍 Utility helpers
# -------------------------------------------------------------------

CARDINALS = ["northern", "southern", "eastern", "western", "central"]

def remove_cardinal_prefix(text: str) -> str:
    """
    Remove leading cardinal adjectives (e.g., 'Western Europe' -> 'Europe').
    """
    pattern = rf"^\s*({'|'.join(CARDINALS)})\s+(.+)$"
    match = re.match(pattern, text.strip(), flags=re.IGNORECASE)
    if match:
        return match.group(2).strip()
    return text.strip()

def is_valid_osm_result(raw: Optional[Dict]) -> bool:
    """
    Validate OSM result to filter out irrelevant classes and low-importance matches.
    """
    if not raw:
        return False

    # Reject non-geographic or local features
    invalid_classes = {
        "amenity", "shop", "office", "highway", "leisure",
        "building", "railway", "waterway", "aeroway", "man_made",
        "craft", "place_of_worship"
    }

    osm_class = raw.get("class", "")
    if osm_class in invalid_classes:
        return False

    # Check minimum importance threshold (heuristic)
    importance = float(raw.get("importance", 0.0))
    if importance < 0.2:
        return False

    # Reject very local features
    place_rank = int(raw.get("place_rank", 0))
    if place_rank > 25:  # high rank = small/local feature
        return False

    return True


# -------------------------------------------------------------------
# 🌍 Utility helpers
# -------------------------------------------------------------------

CARDINALS = ["northern", "southern", "eastern", "western", "central"]
ISO_COUNTRY_CODES = {
    "US", "UK", "GB", "FR", "ES", "IT", "DE", "CN", "IN", "CA",
    "BR", "MX", "RU", "JP", "AU", "CH", "SE", "NO", "FI", "PL", "PT"
}

def remove_cardinal_prefix(text: str) -> str:
    """
    Remove leading cardinal adjectives (e.g., 'Western Europe' -> 'Europe').
    If nothing remains (e.g., 'Northern'), return empty string.
    """
    text = text.strip()
    pattern = rf"^\s*({'|'.join(CARDINALS)})\b\s*(.*)$"
    match = re.match(pattern, text, flags=re.IGNORECASE)
    if match:
        remainder = match.group(2).strip()
        return remainder if remainder else ""
    return text


def merge_adjacent_entities(entities: list[dict]) -> list[dict]:
    """
    Merge consecutive entities of the same type when their character distance <= 1.
    Keeps start/end boundaries consistent.
    """
    if not entities:
        return entities

    # Sort by start position (safety)
    entities = sorted(entities, key=lambda e: e.get("start", 0))
    merged = [entities[0]]

    for e in entities[1:]:
        last = merged[-1]
        if (
            e.get("entity") == last.get("entity") == "geo"
            and e.get("start") - last.get("end", 0) <= 1
        ):
            # Merge contiguous entities
            new_text = (last["text"] + " " + e["text"]).strip()
            merged[-1] = {
                **last,
                "text": new_text,
                "end": e["end"],
                "score": (last.get("score", 1) + e.get("score", 1)) / 2,
            }
        else:
            merged.append(e)

    return merged



# -------------------------------------------------------------------
# 🌍 EntityLinker Class (Nominatim)
# -------------------------------------------------------------------

class EntityLinker:
    """
    Lightweight geolocation linker using OSM Nominatim with LRU cache.
    """

    EXTRATAG_KEYS = ["wikidata", "wikipedia"]

    def __init__(
        self,
        user_agent: str = "siris_geotag_linker",
        cache_maxsize: int = 5000,
        cache_ttl: Optional[float] = None,  # seconds
        importance_threshold: float = 0.0,
        language: str = "en",
        sleep_between_calls: float = 1.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.geocoder = Nominatim(user_agent=user_agent)
        self.cache_maxsize = cache_maxsize
        self.cache_ttl = cache_ttl
        self.importance_threshold = importance_threshold
        self.language = language
        self.sleep = sleep_between_calls
        self._cache: OrderedDict[str, tuple[float, dict | None]] = OrderedDict()

    # ---------------- Cache helpers ----------------
    def _make_key(self, entity: str) -> str:
        e = (entity or "").strip().lower()
        return f"{e}|lang={self.language}"

    def _cache_get(self, key: str):
        if key not in self._cache:
            return None
        ts, value = self._cache[key]
        if self.cache_ttl and (time.time() - ts) > self.cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key, last=True)
        return value

    def _cache_set(self, key: str, value):
        while len(self._cache) >= self.cache_maxsize:
            self._cache.popitem(last=False)
        self._cache[key] = (time.time(), value)
        self._cache.move_to_end(key, last=True)

    # ---------------- Main API ----------------
    def link_entity(self, entity_text: str) -> Optional[Dict]:
        """
        Link a single entity via OSM Nominatim (with caching).
        """
        key = self._make_key(entity_text)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        try:
            res = self.geocoder.geocode(
                entity_text,
                addressdetails=True,
                extratags=True,
                language=self.language,
                timeout=10
            )
        except (GeocoderServiceError, GeocoderTimedOut) as e:
            self.logger.warning(f"⚠️ Nominatim error for '{entity_text}': {e}")
            self._cache_set(key, None)
            return None

        if res is None:
            self._cache_set(key, None)
            return None

        raw = res.raw
        # Apply validation filter
        if not is_valid_osm_result(raw):
            self.logger.debug(f"❌ Filtered out invalid OSM result for '{entity_text}' (class={raw.get('class')})")
            self._cache_set(key, None)
            return None

        self._cache_set(key, raw)
        if self.sleep:
            time.sleep(self.sleep)
        return raw


# -------------------------------------------------------------------
# 🧭 GeoTagLinker Pipeline
# -------------------------------------------------------------------

class GeoTagLinker:
    """
    Geographical entity linker: normalizes and geocodes entities using Nominatim.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.linker = EntityLinker(logger=self.logger)

    def normalise_geographical_entity(self, entity: str) -> str:
        """
        Normalize demonyms/adjectives and remove cardinal prefixes.
        """
        entity = re.sub(
            pattern,
            lambda match: base_transformed_dict.get(match.group(0), match.group(0)),
            entity,
            flags=re.IGNORECASE,
        )
        entity = remove_cardinal_prefix(entity)
        return entity.strip()

    def process_file(self, input_path: str, output_path: str):
        """
        Read geotagging JSONL file, link entities, and write enriched output.
        Keeps only unique entities per section for target roles.
        """
        self.logger.info(f"📂 Processing {input_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        keep_roles = {"Object of study", "Location of research"}
        seen_entities = set()

        with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc=f"Linking {os.path.basename(input_path)}"):
                record = json.loads(line)
                section_id = record.get("section_id")
                entities = record.get("entities", [])

                # 🧩 Merge adjacent entities before processing
                entities = merge_adjacent_entities(entities)
                linked_entities = []

                for ent in entities:
                    role = ent.get("role")
                    text = ent.get("text", "").strip()
                    if not text or role not in keep_roles:
                        continue

                    norm_text = self.normalise_geographical_entity(text)
                    if not norm_text:
                        continue

                    # 🚫 Skip standalone cardinal words like "North", "South", etc.
                    if norm_text.lower() in CARDINALS:
                        self.logger.debug(f"Skipping standalone cardinal: '{norm_text}'")
                        continue

                    # 🚫 Skip short non-ISO codes
                    if len(norm_text) <= 2 and norm_text.upper() not in ISO_COUNTRY_CODES:
                        continue

                    key = norm_text.lower()
                    if key in seen_entities:
                        continue
                    seen_entities.add(key)

                    osm_data = self.linker.link_entity(norm_text)
                    if osm_data:
                        ent["osm"] = osm_data
                        ent["normalized_text"] = norm_text
                        linked_entities.append(ent)


                if linked_entities:
                    out_record = {"section_id": section_id, "entities": linked_entities}
                    fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

        self.logger.info(f"✅ Linked entities saved to {output_path}")


# -------------------------------------------------------------------
# 🚀 CLI entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.utils.logger import setup_logger  # adjust import to your utils if needed

    parser = argparse.ArgumentParser(description="Geotag entity linker using OSM Nominatim.")
    parser.add_argument("--input_dir", required=True, help="Directory with geotagging JSONL outputs")
    parser.add_argument("--output_dir", required=True, help="Directory to save linked outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log = setup_logger(args.output_dir, "geotag_linker")

    linker = GeoTagLinker(logger=log)

    jsonl_files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".jsonl")]
    for path in jsonl_files:
        out_path = os.path.join(
            args.output_dir,
            os.path.basename(path).replace(".jsonl", "_linked.jsonl")
        )

        if os.path.exists(out_path):
            log.info(f"⏭️ Skipping already linked file: {os.path.basename(out_path)}")
            continue

        linker.process_file(path, out_path)

    log.info(f"🔎 Found {len(jsonl_files)} geotagging files to process.")

    for path in jsonl_files:
        out_path = os.path.join(args.output_dir, os.path.basename(path).replace(".jsonl", "_linked.jsonl"))
        linker.process_file(path, out_path)

    log.info("🎉 All geotag files processed successfully.")
