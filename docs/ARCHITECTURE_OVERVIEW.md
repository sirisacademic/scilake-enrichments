# SciLake NER & Entity Linking - Architecture Overview

## System Architecture

The SciLake pipeline is a two-stage system for extracting and linking domain-specific entities from scientific literature. It supports multiple input formats: NIF/RDF files, Title/Abstract JSON, and Legal Text JSON.

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: NIF/RDF Files (.ttl)                   │
│                     Scientific Papers/Documents                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 1: Named Entity Recognition               │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                     │
│  1. NIF Parser → Extract text + structure                       │
│  2. Acronym Expansion → Schwartz-Hearst algorithm (SciSpacy)    │
│  3. GazetteerLinker → Extract + Link (non-cancer domains only)  │
│  4. Neural NER:                                                  │
│     • GLiNER (multi-label semantic matching)                    │
│     • RoBERTa (domain-specific fine-tuned)                      │
│     • AIOner (biomedical - cancer domain)                       │
│  5. Entity Merging → Deduplicate & resolve overlaps             │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│               OUTPUT: Detected Entities (.jsonl)                 │
│                                                                  │
│  Gazetteer entities: Already linked (linking: {...})            │
│  NER entities: Not yet linked (linking: null)                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STAGE 2: Entity Linking (NEL)                   │
├─────────────────────────────────────────────────────────────────┤
│  Links entities NOT already linked by GazetteerLinker           │
│  Configuration loaded from domain el_config                     │
│                                                                  │
│  Linker Options:                                                │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ FTS5Linker ⭐ (cancer domain)            │                   │
│  │   • SQLite FTS5 exact matching           │                   │
│  │   • Per-entity-type indices              │                   │
│  │   • Disk-based, scales to millions       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ SemanticLinker                           │                   │
│  │   • Embedding similarity                 │                   │
│  │   • Fast, fuzzy matching                 │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ InstructLinker                           │                   │
│  │   • Instruction-tuned embeddings         │                   │
│  │   • Better context understanding         │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  ┌──────────────────────────────────────────┐                   │
│  │ RerankerLinker ⭐ (Default for non-cancer)│                   │
│  │   • Stage 1: Embedding retrieval         │                   │
│  │   • Stage 2: LLM reranking               │                   │
│  │   • Can REJECT non-domain entities       │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  Post-Linking Validation:                                       │
│  ┌──────────────────────────────────────────┐                   │
│  │ TypeMatcher                              │                   │
│  │   • Validates NER type matches taxonomy  │                   │
│  │   • Flags type mismatches                │                   │
│  └──────────────────────────────────────────┘                   │
│                                                                  │
│  Features:                                                       │
│  • Context extraction (sentences or token windows)              │
│  • Cache system (persistent, grows over time)                   │
│  • Checkpointing (resume from interruptions)                    │
│  • Batch processing (configurable batch size)                   │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│            OUTPUT: Linked Entities (.jsonl + .ttl)               │
│                                                                  │
│  {                                                               │
│    "text": "wind turbines",                                      │
│    "entity": "energytype",                                       │
│    "start": 42,                                                  │
│    "end": 55,                                                    │
│    "model": "RoBERTa",                                           │
│    "linking": {                                                  │
│      "taxonomy_id": "230000",                                    │
│      "label": "Wind energy",                                     │
│      "source": "IRENA",                                          │
│      "wikidata": "Q43302",                                       │
│      "score": 0.87,                                              │
│      "method": "reranker"                                        │
│    }                                                             │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Domain-Specific Architectures

The pipeline uses different architectures depending on domain characteristics:

### Non-Cancer Domains (Energy, Neuro, CCAM, Maritime)

```
NER Step:
  ┌─────────────────┐     ┌─────────────────┐
  │ GazetteerLinker │     │   Neural NER    │
  │ (FlashText)     │     │ (GLiNER/RoBERTa)│
  │                 │     │                 │
  │ Extracts AND    │     │ Extracts only   │
  │ links entities  │     │ (no linking)    │
  └────────┬────────┘     └────────┬────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌────────────────┐
              │  Merge Results │
              │  (Gaz priority)│
              └────────┬───────┘
                       │
EL Step:               ▼
              ┌────────────────┐
              │ Link unlinked  │
              │ entities via   │
              │ RerankerLinker │
              │ (from el_config)│
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Type Matching  │
              │ Validation     │
              └────────────────┘
```

**Why this works:** Small/medium taxonomies (~9K-50K entries) with unambiguous terms.

### Cancer Domain

```
NER Step:
              ┌─────────────────┐
              │   Neural NER    │
              │    (AIOner)     │
              │                 │
              │ Extracts only   │
              │ (no Gazetteer)  │
              └────────┬────────┘
                       │
EL Step:               ▼
              ┌────────────────┐
              │  FTS5Linker    │
              │ (per entity    │
              │  type indices) │
              └────────────────┘
```

**Why this works:** Large vocabularies (millions of entries) with ambiguous terms (gene symbols like "MET", "ALL", "CAT"). Gazetteer would produce too many false positives scanning text.

---

## Component Details

### 1. NER Stage Components

#### **NIF Parser** (`nif_reader.py`)
- Parses NIF/RDF format (.ttl files)
- Extracts document structure and text
- Preserves character offsets for accurate entity positioning

#### **Title/Abstract Reader** (`title_abstract_reader.py`)
- Parses JSON/JSONL files with publication metadata (oaireid, titles, abstracts)
- **Combined mode (default):** Merges title and abstract into single section
- **Separate mode:** Creates separate sections for title and abstract
- Normalizes whitespace (removes embedded newlines)
- Benefits: Halves section count, better context, faster processing

#### **Legal Text Reader** (`legal_text_reader.py`)
- Parses JSON/JSONL files with legal documents (rsNr, en_lawTitle, en_lawText)
- Combines title and text content
- Normalizes whitespace throughout
- Handles very long documents (chunking done in NER step)

#### **Acronym Expansion** (via `abbreviations` package)
- Uses Schwartz-Hearst algorithm
- Processes per section for consistency
- Example: "PV" → "photovoltaic"

#### **GazetteerLinker** (`gazetteer_linker.py`) - Extraction + Linking

**Purpose:** Scans text during NER step to find AND link taxonomy terms.

- FlashText-based in-memory matching
- Uses taxonomy terms + Wikidata aliases
- **Runs during NER step** (not EL step)
- Both extracts and links in one operation
- Zero false positives on matches
- ⚠️ Known issues: Offset bugs with special characters, memory issues at scale (~300+ files)

**Used by:** Non-cancer domains (Energy, Neuro, CCAM, Maritime)

#### **Neural NER Models**

**GLiNER** (multi-label semantic):
- Uses semantic similarity for classification
- Multi-label config essential for ambiguous entities
- Example labels: `["energy technology", "energy storage", "transportation"]`
- Gives model options → better accuracy

**RoBERTa** (domain-specific):
- Fine-tuned on domain corpus
- Token-level classification
- Fixed output labels per model

**AIOner** (biomedical):
- Specialized for cancer/biology domain
- Detects genes, diseases, species, cell lines

### 2. Entity Linking Stage Components

#### **Configuration via el_config**

Entity linking parameters are centralized in `domain_models.py` under the `el_config` section:

```python
"energy": {
    "el_config": {
        "taxonomy_path": "taxonomies/energy/IRENA.tsv",
        "taxonomy_source": "IRENA",
        "linker_type": "reranker",
        "el_model_name": "intfloat/multilingual-e5-large-instruct",
        "threshold": 0.80,
        "context_window": 5,
        "max_contexts": 5,
        "use_sentence_context": False,
        "reranker_llm": "Qwen/Qwen3-1.7B",
        "reranker_top_k": 7,
        "reranker_fallbacks": True,
    },
}
```

CLI arguments override domain config when specified.

#### **FTS5Linker** (`fts5_linker.py`) - Linking Only

**Purpose:** Link entities already extracted by NER (cancer domain).

- SQLite FTS5 full-text search
- Disk-based (no memory issues)
- Per-entity-type indices
- Text normalization (Greek letters, plurals)
- **Runs during EL step only**
- Does NOT scan text

**Used by:** Cancer domain

#### **SemanticLinker** (`semantic_linker.py`) - Linking Only

- Sentence embedding similarity
- Fast but can have false positives
- Good for large-scale, CPU-only environments

#### **InstructLinker** (`instruct_linker.py`) - Linking Only

- Instruction-tuned embeddings
- Better context understanding than SemanticLinker
- No LLM required

#### **RerankerLinker** (`reranker_linker.py`) - Linking Only

**Default for non-cancer domains** (configured in el_config)

**Two-stage architecture:**
1. **Stage 1:** Fast embedding retrieval (top-k candidates)
2. **Stage 2:** LLM reranking (select best or REJECT)

**Key feature:** Can explicitly REJECT entities that don't belong to the domain.

#### **TypeMatcher** (`type_matching.py`) - Post-Linking Validation

Validates that the NER entity type matches the linked taxonomy concept type:

```python
"energy": {
    "enforce_type_match": True,
    "taxonomy_type_column": "type",
    "type_mappings": {
        "Renewables": "energytype",
        "Fossil fuels": "energytype",
        # ...
    },
}
```

- Flags mismatches for review
- Configurable per domain
- Can be disabled with `--no_type_match`

---

## Caching & Checkpointing

### Cache System

```
Entity: "wind turbines" + context hash
        ↓
Cache Key: "wind turbines|ctx_hash"
        ↓
Cache Hit → Return stored result
Cache Miss → Compute → Store → Return
```

**Benefits:**
- Avoids redundant LLM calls (~70-80% hit rate after warm-up)
- Persistent across runs (JSON file)
- Grows over time with repeated entities

**File:** `outputs/<domain>/el/cache/linking_cache.json`

### Checkpoint System

```
After each batch:
  1. Save results to output file (append)
  2. Update checkpoint with processed IDs
  3. Flush to disk

On resume:
  1. Load checkpoint
  2. Skip already-processed sections/files
  3. Continue from last position
```

**Files:**
- NER: `outputs/<domain>/ner/checkpoints/processed_sections.json`
- EL: `outputs/<domain>/el/checkpoints/processed.json`

---

## Context Extraction

### Token Windows (Default)

```
Text: "The wind turbines convert kinetic energy into electricity."
Entity: "wind turbines" (positions 4-17)
Context window: 3 tokens

Left context: "The"
Right context: "convert kinetic energy"

Combined: "The wind turbines convert kinetic energy"
```

### Sentence Context

```
Text: "... systems. The wind turbines convert kinetic energy. They are..."
Entity: "wind turbines"

Sentence: "The wind turbines convert kinetic energy."
```

**Configuration:**
```python
context_window=5,           # Tokens around entity
max_contexts=5,             # Max contexts per entity
use_sentence_context=False, # True for sentences
```

---

## domain_models.py Structure

Complete domain configuration example:

```python
"energy": {
    # ===== NER Configuration =====
    "gazetteer": {
        "enabled": True,
        "taxonomy_path": "taxonomies/energy/IRENA.tsv",
        "taxonomy_source": "IRENA",
        "model_name": "IRENA-Gazetteer",
        "default_type": "energytype",
    },
    
    "models": [
        {
            "name": "SIRIS-Lab/SciLake-Energy-roberta-base",
            "type": "roberta",
            "threshold": 0.85,
            "output_labels": ["EnergyType", "EnergyStorage"],
        },
    ],
    
    # ===== Entity Filtering =====
    "min_mention_length": 2,
    "blocked_mentions": {"energy", "power", "system", ...},
    
    # ===== Entity Linking Configuration =====
    "linking_strategy": "reranker",
    "el_config": {
        "taxonomy_path": "taxonomies/energy/IRENA.tsv",
        "taxonomy_source": "IRENA",
        "linker_type": "reranker",
        "el_model_name": "intfloat/multilingual-e5-large-instruct",
        "threshold": 0.80,
        "context_window": 5,
        "max_contexts": 5,
        "use_sentence_context": False,
        "reranker_llm": "Qwen/Qwen3-1.7B",
        "reranker_top_k": 7,
        "reranker_fallbacks": True,
    },
    
    # ===== Type Matching Configuration =====
    "enforce_type_match": True,
    "taxonomy_type_column": "type",
    "type_mappings": {
        "Renewables": "energytype",
        "Fossil fuels": "energytype",
        "Energy storage": ["energytype", "energystorage"],
        # ...
    },
}
```

---

## Taxonomy Requirements

### Required Columns

| Column | Required | Description |
|--------|----------|-------------|
| `id` | Yes | Unique identifier |
| `concept` | Yes | Primary label |
| `type` | Recommended | Category for type matching |
| `wikidata_id` | Optional | Wikidata entity ID |
| `wikidata_aliases` | Optional | Pipe-separated aliases |
| `description` | Optional | Concept description (helps LLM) |
| `parent_id` | Optional | For hierarchy |

### Example TSV

```tsv
id	concept	type	wikidata_id	wikidata_aliases	description
230000	Wind energy	Renewables	Q43302	wind power|wind turbines	Wind energy is the conversion...
240110	Solar cell	Renewables	Q15171558	PV cell|photovoltaic	A solar cell converts...
```

---

## CLI Configuration

### Simplified Usage (Recommended)

```bash
# Uses all settings from domain el_config
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step all \
    --resume
```

### Override Specific Settings

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --threshold 0.75 \       # Override el_config
    --reranker_top_k 10 \    # Override el_config
    --resume
```

### Available EL Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--linker_type` | `semantic` \| `instruct` \| `reranker` \| `fts5` | From el_config |
| `--threshold` | Similarity threshold | From el_config (0.80) |
| `--context_window` | Context tokens | From el_config (5) |
| `--max_contexts` | Max contexts | From el_config (5) |
| `--el_model_name` | Embedding model | From el_config |
| `--reranker_llm` | LLM model | From el_config |
| `--reranker_top_k` | Candidates | From el_config (7) |
| `--reranker_fallbacks` | Add fallbacks | From el_config |
| `--reranker_thinking` | Enable CoT | False |
| `--no_type_match` | Disable type matching | Flag |

---

## Long Text Handling

### NER Step

Long texts are handled automatically by chunking:
- Texts split into 512-token chunks with 50-token overlap
- Entities deduplicated across chunks based on (entity_type, start, end)
- No length limit

### EL Step

SpaCy has a 1M character limit for context extraction:
```python
MAX_SECTION_LENGTH = 1000000
if len(section_text) > MAX_SECTION_LENGTH:
    logger.warning(f"⚠️ Truncating section {section_id}")
    section_text = section_text[:MAX_SECTION_LENGTH]
```

**Impact:** Only affects ~0.04% of documents. Entities beyond truncation point get linked without context.

---

## Incremental Saving

For title/abstract and legal text formats, results are saved incrementally after each batch:

```python
# After each batch of 1000 sections:
with open(out_path, 'a', encoding='utf-8') as f:
    for record in batch_results:
        f.write(json.dumps(record) + '\n')
save_json(processed, checkpoint_file)
```

**Benefits:**
- Results available immediately (don't wait for completion)
- No data loss on crash
- Safe to stop and resume at any time

---

## Parallel Processing

For large datasets (millions of records), split input files and run in parallel:

```bash
# Split into 6 parts
split -n l/6 -d --additional-suffix=.json input.json input_part

# Run NER in parallel
for i in 00 01 02 03 04 05; do
    nohup python src/pipeline.py \
        --domain energy --step ner --input_format title_abstract \
        --input input_part${i}.json --output outputs/part${i} --resume \
        > outputs/part${i}_ner.log 2>&1 &
done

# Run EL in parallel (uses el_config)
for i in 00 01 02 03 04 05; do
    nohup python src/pipeline.py \
        --domain energy --step el \
        --output outputs/part${i} --resume \
        > outputs/part${i}_el.log 2>&1 &
done

# Merge results
cat outputs/part*/el/*.jsonl > outputs/merged/el/merged.jsonl
```

### GPU Memory Planning

| Instances | GPU Memory | RTX 4000 (20GB) | RTX 6000 (49GB) |
|-----------|------------|-----------------|-----------------|
| 1 | ~5-6GB | ✅ | ✅ |
| 3 | ~15-18GB | ✅ | ✅ |
| 6 | ~30-36GB | ❌ | ✅ |

---

## Design Principles

### 1. **Separation of Concerns**

- NER detects entities → EL links them
- Each linker is independent and swappable
- Cache layer decouples from linking logic
- Configuration centralized in el_config

### 2. **Fail-Safe Architecture**

- Checkpointing at file level
- Cache persisted to disk
- Resume from any interruption
- Graceful degradation (no linking is better than wrong linking)

### 3. **Performance Optimization**

- Cache-first strategy (avoids redundant computation)
- Batch processing with progress tracking
- Two-stage linking (fast retrieval + accurate reranking)
- Disk-based storage for large vocabularies (FTS5)

### 4. **Domain Agnostic**

- Same architecture for all domains
- Domain-specific configs in `src/domain_models.py`
- Taxonomy-driven (not hardcoded rules)
- Flexible prompt templates

---

## Quality Metrics

### Target Performance

| Metric | Target | Typical (Energy) |
|--------|--------|------------------|
| NER Precision | >90% | ~92% |
| NER Recall | >85% | ~87% |
| Linking Precision | >90% | ~93% (Reranker) |
| Linking Rate | >80% | ~85% |
| Cache Hit Rate | >70% (after 100 docs) | ~80% |
| Throughput | >100 entities/sec | ~150 entities/sec (warm) |

### Evaluation Strategy

1. **Manual Annotation** (sample 100-200 entities)
   - Check NER accuracy (correct spans + labels)
   - Check linking accuracy (correct taxonomy IDs)
   - Identify systematic errors

2. **Statistical Analysis**
   - Linking rate by entity type
   - Score distribution (helps set threshold)
   - Cache efficiency over time

3. **Error Analysis**
   - False positives (wrong links)
   - False negatives (missed links)
   - Systematic biases (e.g., always linking to broad categories)

---

## Logging & Monitoring

### Log Structure

```
outputs/<domain>/logs/<domain>_el.log

2025-11-07 10:00:00 [INFO] 🔗 Starting Entity Linking for domain=energy
2025-11-07 10:00:00 [INFO] Threshold: 0.8
2025-11-07 10:00:00 [INFO] Reranker: llm=Qwen/Qwen3-1.7B, top_k=7
2025-11-07 10:00:00 [INFO] ✅ TypeMatcher initialized: 14 type mappings
2025-11-07 10:01:40 [INFO] ✅ Taxonomy index ready: 8947 entries
2025-11-07 10:01:45 [DEBUG] ✅ 'wind turbines' → 'Wind energy' (score=0.87)
2025-11-07 10:01:46 [DEBUG] ❌ 'emissions' → REJECTED (not energy concept)
2025-11-07 10:01:46 [DEBUG] 📊 Cache: 1 hit, 2 misses | Links added: 2/3
2025-11-07 10:05:00 [INFO] ✅ paper1.jsonl: 45/52 entities linked (86.5%)
2025-11-07 10:10:00 [INFO] 💾 Cache checkpoint: 5234 entries saved
2025-11-07 11:00:00 [INFO] 📊 Cache stats: 4456 linked (85.1%), 778 rejected
2025-11-07 11:00:00 [INFO] 🎉 Entity Linking complete!
```

### Progress Tracking

Uses `tqdm` for visual progress:

```
Processing files: 100%|████████████| 1000/1000 [00:45:23<00:00, 22.1 files/s]
Linking entities: 89%|████████▉  | 45234/51000 [00:12:34<00:01:23, 360.5 ent/s]
```

---

## Summary

The SciLake pipeline provides:

✅ **Multiple Input Formats**: NIF, Title/Abstract, Legal Text  
✅ **Flexible NER**: Multiple models for high recall  
✅ **Entity Filtering**: Domain-level blocked mentions and min length  
✅ **Advanced Linking**: Five linking strategies, from fast to accurate  
✅ **Centralized Configuration**: EL parameters in domain el_config  
✅ **Type Validation**: TypeMatcher validates NER-taxonomy type consistency  
✅ **Production-Ready**: Checkpointing, caching, incremental saving, logging  
✅ **Domain-Agnostic**: Easy to adapt to new domains  
✅ **High Quality**: >90% precision, >85% linking rate  
✅ **Scalable**: Parallel processing for millions of documents  
✅ **Memory-Safe**: FTS5 for large vocabularies without OOM

**Recommended Configuration**: 
- **Exact matching**: FTS5Linker (disk-based, production-ready)
- **Semantic matching**: RerankerLinker with entity-only retrieval (default via el_config)

---

## Additional Documentation

For more detailed information, see:

- **[README.md](README.md)** - Quick start and overview
- **[ENTITY_LINKING_README.md](ENTITY_LINKING_README.md)** - Detailed guide to all 5 linking approaches
- **[RERANKER_GUIDE.md](RERANKER_GUIDE.md)** - Deep dive into RerankerLinker (recommended approach)
- **[CONFIGURATION_GUIDE.md](CONFIGURATION_GUIDE.md)** - Configuration recipes and best practices
