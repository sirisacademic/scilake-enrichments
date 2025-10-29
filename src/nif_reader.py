import os
import re
import pandas as pd
from rdflib import Graph, Namespace
from tqdm import tqdm
import spacy
from scispacy.abbreviation import AbbreviationDetector
from src.utils.logger import setup_logger


# --------------------------------------------------
# ✅ SciSpacy Setup (lazy load)
# --------------------------------------------------
_nlp = None

def get_nlp():
    """Lazy-load SciSpacy model for acronym detection."""
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
        _nlp.add_pipe("abbreviation_detector")
    return _nlp


# --------------------------------------------------
# RDF Namespaces
# --------------------------------------------------
NIF = Namespace("http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#")
DCT = Namespace("http://purl.org/dc/terms/")
SCILAKE = Namespace("http://scilake-projekt.eu/ontology/")  # fixed version


# --------------------------------------------------
# File helpers
# --------------------------------------------------
def parse_file_metadata(path: str):
    """Extract journal and IDs from filenames like journal::uuid1::uuid2.ttl"""
    fname = os.path.basename(path)
    match = re.match(r"(.+?)::([a-f0-9]+)::([a-f0-9]+)\.ttl$", fname)
    if match:
        return match.groups()
    return ("unknown", "unknown", "unknown")


# --------------------------------------------------
# RDF parsing
# --------------------------------------------------
def parse_nif_file(path: str, logger=None) -> list[dict]:
    """
    Parse a single .ttl file and extract all NIF sections.

    Returns:
        List of dicts with section metadata and text content.
    """
    g = Graph()
    try:
        g.parse(path, format="ttl")
    except Exception as e:
        if logger:
            logger.warning(f"Skipping {path} — parse error: {e}")
        return []

    journal, id1, id2 = parse_file_metadata(path)
    records = []

    for subj in g.subjects(predicate=NIF.anchorOf):
        section_content = str(g.value(subj, NIF.anchorOf, default="")).strip()
        if not section_content:
            continue

        record = {
            "section_id": str(subj),
            "file_path": path,
            "journal": journal,
            "id1": id1,
            "id2": id2,
            "section_type": str(g.value(subj, SCILAKE.DocumentPartType, default="")).strip(),
            "section_number": str(g.value(subj, DCT.section_number, default="")).strip(),
            "section_title": str(g.value(subj, DCT.title, default="")).strip(),
            "section_content": section_content,
        }
        records.append(record)

    # Extract fulltext (nif:Context)
    fulltext_candidates = [
        (str(subj), str(g.value(subj, NIF.isString, default="")).strip())
        for subj in g.subjects(predicate=NIF.isString)
    ]
    if fulltext_candidates:
        fulltext_id, fulltext = max(fulltext_candidates, key=lambda x: len(x[1]))
    else:
        fulltext_id, fulltext = (None, "")

    for r in records:
        r["fulltext_id"] = fulltext_id
        r["fulltext"] = fulltext

    return records


def load_all_nif(base_dir: str, logger=None) -> pd.DataFrame:
    """
    Recursively load all .ttl NIF files under a directory.

    Returns:
        DataFrame of all NIF sections.
    """
    ttl_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(base_dir)
        for f in files if f.endswith(".ttl")
    ]
    if logger:
        logger.info(f"📂 Found {len(ttl_files)} NIF files in {base_dir}")

    all_records = []
    for path in tqdm(ttl_files, desc="Parsing NIF files"):
        all_records.extend(parse_nif_file(path, logger=logger))

    df = pd.DataFrame(all_records)
    if logger:
        logger.info(f"✅ Parsed {len(df)} total NIF sections")
    return df


# --------------------------------------------------
# Acronym detection and expansion
# --------------------------------------------------
def get_acronym_map(fulltext: str) -> dict:
    """Extract acronym → long-form mappings from fulltext using SciSpacy."""
    if not fulltext:
        return {}
    nlp = get_nlp()
    doc = nlp(fulltext)
    return {
        abbr.text: abbr._.long_form.text
        for abbr in doc._.abbreviations
        if abbr._.long_form
    }


def expand_acronyms_in_text(text: str, acronym_map: dict) -> str:
    """Replace acronyms in text using word boundaries to avoid false matches."""
    if not text or not acronym_map:
        return text
    for abbr, full in acronym_map.items():
        # Use regex word boundaries to avoid partial matches
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text)
    return text


def apply_acronym_expansion(df: pd.DataFrame, logger=None) -> pd.DataFrame:
    """
    Apply acronym expansion to each section using fulltext from same file.

    Args:
        df: DataFrame from `load_all_nif()`
    Returns:
        DataFrame with `section_content_expanded`
    """
    expanded = []
    for path, group in tqdm(df.groupby("file_path"), desc="Expanding acronyms"):
        fulltext = group.iloc[0]["fulltext"]
        acronym_map = get_acronym_map(fulltext)

        group = group.copy()
        group["acronym_map"] = [acronym_map] * len(group)
        group["section_content_expanded"] = group["section_content"].apply(
            lambda t: expand_acronyms_in_text(t, acronym_map)
        )
        expanded.append(group)

        if logger:
            logger.info(f"Expanded acronyms for {os.path.basename(path)} ({len(acronym_map)} found)")

    result = pd.concat(expanded, ignore_index=True)
    result.drop(columns=["fulltext"], inplace=True, errors="ignore")

    if logger:
        logger.info(f"✅ Acronym expansion complete for {len(result)} sections")
    return result


# --------------------------------------------------
# CLI (for debugging)
# --------------------------------------------------
if __name__ == "__main__":
    from utils.logger import setup_logger
    import argparse

    parser = argparse.ArgumentParser(description="Test NIF Reader")
    parser.add_argument("--input", required=True, help="Path to NIF directory")
    args = parser.parse_args()

    logger = setup_logger("logs", "nif_reader")

    df_sections = load_all_nif(args.input, logger=logger)
    df_expanded = apply_acronym_expansion(df_sections, logger=logger)

    logger.info("✅ Sample output:")
    logger.info(df_expanded[["section_id", "section_type", "section_content_expanded"]].head(3).to_string())
