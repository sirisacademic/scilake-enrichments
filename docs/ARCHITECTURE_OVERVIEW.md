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
│  │ RerankerLinker ⭐ (Recommended)          │                   │
│  │   • Stage 1: Embedding retrieval         │                   │
│  │   • Stage 2: LLM reranking               │                   │
│  │   • Can REJECT non-domain entities       │                   │
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
              │ Semantic/      │
              │ Reranker       │
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
- Careful offset handling (tokens ≠ characters)

#### **Entity Merging**
- Resolves overlaps (longest span wins)
- Deduplicates across models
- Preserves provenance (tracks which model found entity)

#### **Entity Filtering** (applied in EL step)

Domain-level filters configured in `domain_models.py`:

```python
"energy": {
    "min_mention_length": 2,  # Skip entities shorter than 2 chars
    "blocked_mentions": {"energy", "power", "system", "data"},  # Skip generic terms
}
```

Supports per-entity-type configuration:
```python
"cancer": {
    "min_mention_length": {"gene": 2, "disease": 3, "_default": 2},
    "blocked_mentions": {
        "species": {"patient", "patients", "man", "woman"},
        "disease": {"pain", "syndrome"},
    }
}
```

---

### 2. Entity Linking Components

#### **Context Extraction**

Two modes available:

**Sentence Context** (recommended):
```python
"Wind turbines convert kinetic energy into electricity."
                →
         Full sentence provides semantic context
```

**Token Window Context**:
```python
"... renewable wind turbines convert kinetic ..."
              ← entity →
      ← 3 tokens      3 tokens →
```

#### **FTS5Linker: Linking Only (EL Step)** ⭐

The FTS5Linker provides production-ready exact matching using SQLite FTS5. Unlike GazetteerLinker, it does **not scan text** - it receives entities already extracted by NER and looks them up.

**Purpose:** Link entities that were extracted by Neural NER (not extraction).

**Used by:** Cancer domain (large, ambiguous vocabularies)

```
1. Pre-build SQLite FTS5 Index:
   python src/build_fts5_indices.py \
       --taxonomy taxonomies/cancer/NCBI_GENE.tsv \
       --output indices/cancer/ncbi_gene.db

2. Matching Strategy (for each entity from NER):
   a. Try exact match on concept column (case-insensitive)
   b. Try exact match on synonyms
   c. Try normalized variants:
      • Greek letters: "ifn-γ" → "ifn-g" → "ifng"
      • Spacing: "erk1 / 2" → "erk1/2"
      • Plurals: "cytokines" → "cytokine"
   d. Disambiguate by frequency if multiple matches
```

**GazetteerLinker vs FTS5Linker:**

| Aspect | GazetteerLinker | FTS5Linker |
|--------|-----------------|------------|
| **Stage** | NER step | EL step |
| **Purpose** | Extraction + Linking | Linking only |
| **Scans text?** | Yes | No |
| **Memory** | High (in-memory) | Low (disk-based) |
| **Large vocabularies** | ❌ OOM risk | ✅ Millions of entries |
| **Text normalization** | ❌ No | ✅ Built-in |
| **Disambiguation** | First match wins | Frequency-based |

#### **Embedding-Based Retrieval**

**Taxonomy Index Building**:
```
Load IRENA.tsv:
  230000 | Wind energy | Q43302 | wind power, wind turbines
         →
Encode all entries:
  encode("passage: Wind energy")          → [768-dim vector]
  encode("passage: wind power")           → [768-dim vector]
  encode("passage: wind turbines")        → [768-dim vector]
         →
Store in memory:
  ~9000 entries × 768 dimensions = ~6 MB
```

**Query Matching**:
```
Entity: "wind turbines"
Context: "Wind turbines convert kinetic energy into electricity."
         →
Encode query:
  query_emb = encode("query: Wind turbines convert kinetic energy...")
         →
Compute similarities:
  scores = query_emb @ taxonomy_embeddings.T
         →
Results:
  1. Wind energy (0.87) → Best match
  2. wind power (0.85)
  3. Solar energy (0.32)
  4. Nuclear energy (0.28)
```

#### **RerankerLinker: Two-Stage Approach**

**Stage 1: Fast Embedding Retrieval** (~10-20ms)

Parameters:
- `use_context_for_retrieval`: Whether to include context in embedding matching
  - `False` (default): Entity text only → prevents context contamination
  - `True`: Entity + context → better semantic matching but risk of false positives

Process:
```python
# Option 1: Entity-only (safer, default)
query = "query: wind turbines"

# Option 2: With context (riskier)
query = "query: Wind turbines convert kinetic energy..."

# Retrieve top-k candidates
candidates = get_top_k_similar(query, k=5)
# Returns: [(taxonomy_id, score), ...]

# Optional: Add top-level fallbacks
if add_fallbacks:
    candidates += top_level_categories
```

**Stage 2: LLM Reranking** (~50-100ms)

Uses local LLM (e.g., Qwen) to validate candidates:

```python
prompt = f"""
You are a {domain} domain expert. Given an entity and its context,
select the best matching concept or REJECT if none fit.

Entity: "{entity_text}"
Context: "{sentence_context}"

Candidates:
1. {label_1} ({taxonomy_id_1}) - {description_1}
2. {label_2} ({taxonomy_id_2}) - {description_2}
...

Instructions:
- Consider the entity text and surrounding context
- Reject if entity is not truly a {domain} concept
- Reject if entity is a chemical, pollutant, or generic term
- Prefer specific matches over broad categories

Answer: [1-{k} or REJECT]
"""

llm_output = query_llm(prompt)
# Returns: "1" or "3" or "REJECT"
```

Key benefits of two-stage approach:
- **Speed**: Embedding retrieval narrows candidates fast
- **Accuracy**: LLM catches nuanced semantic distinctions
- **Safety**: LLM can reject non-domain terms
- **Flexibility**: Works with or without context
- **Domain-agnostic**: Same architecture for all domains

---

## Performance Characteristics

### Processing Speed

| Component | Speed | Notes |
|-----------|-------|-------|
| NIF Parsing | ~100 ms/doc | Depends on doc size |
| Acronym Expansion | ~50 ms/doc | Per-section processing |
| Gazetteer Matching | ~20 ms/doc | FlashText is very fast |
| FTS5 Matching | ~20 ms/doc | SQLite is equally fast |
| GLiNER | ~200 ms/doc | GPU-dependent |
| RoBERTa | ~150 ms/doc | GPU-dependent |
| Semantic Linker | ~10-20 ms/entity | Cached after first run |
| Instruct Linker | ~15-30 ms/entity | Slightly slower than semantic |
| Reranker Linker | ~50-100 ms/entity | LLM reranking overhead |

### Cache Performance

```
Cache Hit Rate Over Time:

100% │                                    ╭───────
     │                           ╭────────╯
 80% │                    ╭──────╯
     │              ╭─────╯
 60% │         ╭────╯
     │    ╭────╯
 40% │ ╭──╯
     │╭╯
 20% │
     │
  0% └─────┬─────┬─────┬─────┬─────┬─────┬─────
        0   100  200   500  1000  2000  5000+ docs

Cache Size Growth:
  First 100 docs:  ~500 entries
  First 1000 docs: ~3000 entries
  First 5000 docs: ~8000 entries (plateaus)
```

### Memory Usage

| Component | Memory | Persistent |
|-----------|--------|-----------|
| Gazetteer (FlashText) | ~50-200 MB | Yes (in RAM) |
| FTS5 indices | ~10-50 MB (disk) | Yes (disk) |
| IRENA embeddings | ~6 MB | Yes (in RAM) |
| Embedding model weights | ~500 MB | Yes (in RAM) |
| LLM model weights | ~3-7 GB | Yes (in RAM/GPU) |
| GLiNER weights | ~500 MB | Yes (in RAM/GPU) |
| RoBERTa weights | ~500 MB | Yes (in RAM/GPU) |
| Linking cache | ~15-30 MB | Yes (disk + RAM) |
| Working memory | ~100 MB | No (transient) |
| **Total (Reranker)** | **~5-8 GB** | Mixed |
| **Total (FTS5 only)** | **~1-2 GB** | Mixed |

---

## File Organization

```
project/
│
├── configs/
│   └── domain_models.py          # Domain-specific model configs
│
├── src/
│   ├── pipeline.py                # Main orchestrator
│   ├── nif_reader.py              # NIF/RDF parser
│   ├── title_abstract_reader.py   # Title/abstract JSON reader
│   ├── legal_text_reader.py       # Legal text JSON reader
│   ├── ner_runner.py              # NER coordinator
│   ├── gazetteer_linker.py        # FlashText exact matching
│   ├── fts5_linker.py             # SQLite FTS5 exact matching ⭐
│   ├── build_fts5_indices.py      # Build FTS5 indices from TSV
│   ├── semantic_linker.py         # Basic embedding linking
│   ├── instruct_linker.py         # Instruction-tuned linking
│   ├── reranker_linker.py         # Two-stage linking ⭐
│   ├── geo_linker.py              # Geographic entity linking
│   ├── geotagging_runner.py       # Geotagging pipeline
│   ├── affilgood_runner.py        # Affiliation enrichment
│   └── utils/
│       ├── io_utils.py            # I/O helpers
│       └── logger.py              # Logging setup
│
├── indices/                       # FTS5 SQLite indices
│   └── <domain>/
│       └── *.db                   # Pre-built FTS5 databases
│
├── taxonomies/
│   └── <domain>/
│       └── *.tsv                  # Taxonomy source files
│
├── data/
│   └── <domain>/
│       └── *.ttl                  # Input NIF files
│
└── outputs/
    └── <domain>/
        ├── ner/                   # NER outputs
        │   ├── *.jsonl            # Detected entities
        │   └── expanded/          # With acronyms expanded
        │       └── *_expanded.csv
        │
        ├── el/                    # Entity Linking outputs
        │   ├── *.jsonl            # Linked entities
        │   └── cache/
        │       └── linking_cache.json  # Persistent cache
        │
        ├── checkpoints/           # Resume points
        │   └── processed.json
        │
        └── logs/                  # Detailed logs
            ├── <domain>_ner.log
            └── <domain>_el.log
```

---

## Data Flow Example

### Input NIF File (`paper1.ttl`)

```turtle
@prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> .

<http://scilake.eu/resource#context_1>
    a nif:Context ;
    nif:isString "Wind turbines convert kinetic energy into electricity." .
```

### After NER (`paper1.jsonl`)

```json
{
  "doc_id": "paper1",
  "entities": [
    {
      "text": "Wind turbines",
      "entity": "energytype",
      "start": 0,
      "end": 13,
      "model": "RoBERTa",
      "confidence": 0.94,
      "linking": null
    },
    {
      "text": "kinetic energy",
      "entity": "energytype",
      "start": 22,
      "end": 36,
      "model": "GLiNER",
      "confidence": 0.89,
      "linking": null
    }
  ]
}
```

### After Entity Linking (`paper1.jsonl`)

```json
{
  "doc_id": "paper1",
  "entities": [
    {
      "text": "Wind turbines",
      "entity": "energytype",
      "start": 0,
      "end": 13,
      "model": "RoBERTa",
      "confidence": 0.94,
      "linking": {
        "taxonomy_id": "230000",
        "label": "Wind energy",
        "source": "IRENA",
        "wikidata": "Q43302",
        "score": 0.87,
        "method": "reranker",
        "context": "Wind turbines convert kinetic energy into electricity.",
        "candidates_considered": 5
      }
    },
    {
      "text": "kinetic energy",
      "entity": "energytype",
      "start": 22,
      "end": 36,
      "model": "GLiNER",
      "confidence": 0.89,
      "linking": null  // Rejected by reranker (too generic)
    }
  ]
}
```

### Output NIF File (enriched, `.ttl`)

```turtle
@prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> .
@prefix itsrdf: <http://www.w3.org/2005/11/its/rdf#> .

<http://scilake.eu/resource#offset_0_13>
    a nif:EntityOccurrence ;
    nif:referenceContext <http://scilake.eu/resource#context_1> ;
    nif:beginIndex "0"^^xsd:int ;
    nif:endIndex "13"^^xsd:int ;
    nif:anchorOf "Wind turbines" ;
    itsrdf:taIdentRef <http://irena.org/kb/230000> ;
    itsrdf:taIdentRef <http://www.wikidata.org/entity/Q43302> .
```

---

## Configuration Patterns

### For High Precision (avoid false positives)

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type reranker \
    --threshold 0.8 \
    --use_context_for_retrieval false \  # Entity-only retrieval
    --reranker_top_k 3 \                 # Fewer candidates
    --context_window 3
```

### For High Recall (maximize linking rate)

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type reranker \
    --threshold 0.6 \
    --use_context_for_retrieval true \   # Context helps find more matches
    --reranker_top_k 10 \                # More candidates
    --reranker_fallbacks \               # Include broad categories
    --context_window 5
```

### For Speed (large-scale processing)

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type semantic \             # Fastest option
    --threshold 0.7 \
    --context_window 3
```

### For Production Exact Matching (FTS5)

Configure in `domain_models.py`:

```python
"energy": {
    "gazetteer": {"enabled": False},
    "linking_strategy": "fts5",
    "fts5_linkers": {
        "energytype": {
            "index_path": "indices/energy/irena.db",
            "taxonomy_source": "IRENA",
        }
    }
}
```

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

# Run in parallel
for i in 00 01 02 03 04 05; do
    nohup python src/pipeline.py \
        --domain energy --step ner --input_format title_abstract \
        --input input_part${i}.json --output outputs/part${i} --resume \
        > outputs/part${i}.log 2>&1 &
done

# Merge results
cat outputs/part*/ner/*.jsonl > outputs/merged/ner/merged.jsonl
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
- Domain-specific configs in `configs/domain_models.py`
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
✅ **Production-Ready**: Checkpointing, caching, incremental saving, logging  
✅ **Domain-Agnostic**: Easy to adapt to new domains  
✅ **High Quality**: >90% precision, >85% linking rate  
✅ **Scalable**: Parallel processing for millions of documents  
✅ **Memory-Safe**: FTS5 for large vocabularies without OOM

**Recommended Configuration**: 
- **Exact matching**: FTS5Linker (disk-based, production-ready)
- **Semantic matching**: RerankerLinker with entity-only retrieval for optimal precision/recall balance

---

## Additional Documentation

For more detailed information, see:

- **[README.md](README.md)** - Quick start and overview
- **[ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)** - Detailed guide to all 5 linking approaches
- **[RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)** - Deep dive into RerankerLinker (recommended approach)
- **[CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)** - Configuration recipes and best practices
