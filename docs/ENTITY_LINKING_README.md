# Entity Linking - Detailed Guide

Comprehensive guide to all entity linking approaches in the SciLake pipeline.

---

## Overview

Entity Linking (NEL) maps detected entities to concepts in a controlled vocabulary (taxonomy). The pipeline offers **five linking strategies**, each with different trade-offs between speed, accuracy, and complexity.

### Important: Two Types of Linkers

The pipeline uses linkers at **two different stages**:

| Stage | Component | Purpose | Domains |
|-------|-----------|---------|---------|
| **NER Step** | GazetteerLinker | Extraction + Linking (finds entities AND links them) | Energy, Neuro, CCAM, Maritime |
| **EL Step** | FTS5Linker, SemanticLinker, etc. | Linking only (links entities already found by NER) | Cancer (FTS5), All (Semantic/Reranker) |

This distinction is important: **GazetteerLinker scans text to find entities**, while **FTS5Linker and other EL linkers receive entities already extracted by NER**.

---

## Quick Comparison

### Extraction + Linking (NER Step)

| Linker | Purpose | Speed | Memory | Use Case |
|--------|---------|-------|--------|----------|
| **GazetteerLinker** | Scan text for taxonomy matches | ⚡⚡⚡ | High (in-memory) | Small/medium taxonomies |

### Linking Only (EL Step)

| Linker | Speed | Accuracy | GPU | Memory | Use Case |
|--------|-------|----------|-----|--------|----------|
| **FTS5** ⭐ | ⚡⚡⚡ Instant | 🎯🎯🎯 100% precision | No | Low (disk) | Large vocabularies, production |
| **Semantic** | ⚡⚡ Fast | 🎯🎯 Good | No | Medium | Large-scale, CPU-only |
| **Instruct** | ⚡ Medium | 🎯🎯🎯 Better | Optional | Medium | Balanced speed/accuracy |
| **Reranker** | 🐢 Slower | 🎯🎯🎯🎯 Best | Yes (LLM) | High | High accuracy needs |

---

## 1. GazetteerLinker (Extraction + Linking)

### Description

FlashText-based exact matching that **scans text** to find taxonomy terms and their aliases. Runs during the **NER step** (not EL step) and both extracts and links entities in a single operation.

### When It Runs

```
NER Step:
  1. Parse NIF file
  2. Expand acronyms
  3. → GazetteerLinker.extract_entities() ← Finds AND links entities
  4. Neural NER (GLiNER/RoBERTa)
  5. Merge results (Gazetteer has priority)
```

### How It Works

```
FlashText Dictionary:
  "wind energy" → IRENA:230000
  "wind power" → IRENA:230000
  "wind turbines" → IRENA:230000
  "solar cell" → IRENA:240110
  ...

Input text: "Wind turbines convert energy..."
              ↓ (exact match)
Linked: "Wind turbines" → IRENA:230000
```

### Configuration

Enabled in `configs/domain_models.py`:

```python
DOMAIN_MODELS = {
    "energy": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "case_sensitive": False
        }
    }
}
```

### Pros & Cons

**Pros:**
- ✅ Zero false positives (100% precision)
- ✅ Instant speed (~1ms per document)
- ✅ No model weights required
- ✅ Works offline

**Cons:**
- ❌ Limited recall (only exact matches)
- ❌ No fuzzy matching
- ❌ Sensitive to spelling variations
- ❌ Misses paraphrases ("wind turbine" vs "turbine for wind")
- ❌ In-memory: High RAM usage for large vocabularies
- ❌ Known issues with FlashText offset calculation for special characters
- ❌ May cause segmentation faults with pandas C parser during large-scale processing

### When to Use

- ✅ Small to medium taxonomies (<50k concepts)
- ✅ Text uses standardized terminology
- ✅ Processing speed is critical
- ✅ You need guaranteed precision

> **Note:** For large-scale production processing, consider using **FTS5 Linker** instead to avoid memory issues.

---

## 2. FTS5Linker ⭐ (Linking Only - Recommended for Production)

### Description

SQLite FTS5-based exact matching designed for **linking entities that were already extracted by NER**. Unlike GazetteerLinker, FTS5Linker does **not scan text** - it receives entity mentions and looks them up in a disk-based index. This makes it suitable for very large vocabularies (millions of entries) and large-scale processing without memory issues.

### Important Distinction from GazetteerLinker

| Aspect | GazetteerLinker | FTS5Linker |
|--------|-----------------|------------|
| **Input** | Full document text | Single entity mention |
| **Output** | List of entities with positions + links | Taxonomy match or None |
| **When** | NER step | EL step |
| **Scans text?** | Yes | No |
| **Memory** | High (in-memory FlashText) | Low (disk-based SQLite) |

### Why FTS5 Was Developed

The FTS5Linker was developed to address several issues encountered with the FlashText-based Gazetteer during large-scale processing:

1. **Memory Issues**: When processing hundreds or thousands of files, the combination of in-memory FlashText dictionaries and pandas CSV loading led to memory fragmentation and segmentation faults
2. **Pandas C Parser Crashes**: The pipeline would crash after ~310 files due to pandas' C parser buffer allocation issues with very long text fields (5000+ characters)
3. **FlashText Offset Bugs**: FlashText occasionally returns incorrect character offsets when text contains special characters

FTS5 solves these by using a disk-based database backend that doesn't suffer from memory fragmentation.

### How It Works

```
1. Pre-build SQLite FTS5 Index:
   CREATE VIRTUAL TABLE entities_fts USING fts5(concept, synonyms, ...);
   
   Table entries:
     id: "NCBI:7157", concept: "TP53", synonyms: "p53|tumor protein p53"
     id: "NCBI:1956", concept: "EGFR", synonyms: "ErbB-1|HER1"
     ...

2. For Each Entity:
   Query: "TP53" (case-insensitive)
   
3. Matching Strategy:
   a. Try exact match on concept column
   b. Try exact match on synonyms
   c. Try normalized variants (Greek letters, spacing, plurals)
   
4. Disambiguation:
   If multiple matches → return highest frequency entry

Example:
  Input: "ifn-γ"
  Normalized: "ifn-g" → "ifng"
  Match: "IFNG" (NCBI:3458)
```

### Text Normalization

FTS5Linker includes built-in text normalization to improve matching:

| Normalization | Example |
|---------------|---------|
| Greek letters | `ifn-γ` → `ifn-g` |
| Greek aggressive | `ifn-γ` → `ifng` (removes hyphen) |
| Spacing | `erk1 / 2` → `erk1/2` |
| Depluralization | `cytokines` → `cytokine` |

### Configuration

In `configs/domain_models.py`:

```python
DOMAIN_MODELS = {
    "cancer": {
        "gazetteer": {
            "enabled": False,  # Disable FlashText gazetteer
        },
        "linking_strategy": "fts5",
        
        "fts5_linkers": {
            "gene": {
                "index_path": "indices/cancer/ncbi_gene.db",
                "taxonomy_source": "NCBI_Gene",
                "taxonomy_path": "taxonomies/cancer/NCBI_GENE.tsv",
            },
            "species": {
                "index_path": "indices/cancer/ncbi_species.db",
                "taxonomy_source": "NCBI_Taxonomy",
                "blocked_mentions": {
                    "patient", "patients", "man", "woman",
                    # ... other blocked terms
                },
            },
            "disease": {
                "index_path": "indices/cancer/doid_disease.db",
                "taxonomy_source": "DOID",
                "fallback": "semantic",  # Optional semantic fallback
            },
        },
    }
}
```

### Building FTS5 Indices

Use the `build_fts5_indices.py` script to create indices from taxonomy TSV files:

```bash
python src/build_fts5_indices.py \
    --taxonomy taxonomies/cancer/NCBI_GENE.tsv \
    --output indices/cancer/ncbi_gene.db \
    --source NCBI_Gene
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `index_path` | Path to SQLite FTS5 database file |
| `taxonomy_source` | Source name for linking output (e.g., "NCBI_Gene") |
| `taxonomy_path` | Original TSV file (for reference/rebuilding) |
| `blocked_mentions` | Set of lowercase terms to always reject |
| `fallback` | Optional fallback strategy ("semantic") for unmatched entities |

### Pros & Cons

**Pros:**
- ✅ Zero false positives (100% precision on matches)
- ✅ Instant speed (~1ms per entity)
- ✅ Low memory usage (disk-based)
- ✅ Scales to millions of entities
- ✅ No segmentation faults during large-scale processing
- ✅ Built-in text normalization (Greek, spacing, plurals)
- ✅ Frequency-based disambiguation
- ✅ Per-entity-type configuration (different indices for genes, diseases, etc.)
- ✅ Production-ready and battle-tested

**Cons:**
- ❌ Limited recall (only exact matches + normalized variants)
- ❌ Requires pre-building indices
- ❌ No fuzzy matching
- ❌ No semantic understanding

### When to Use

- ✅ Large vocabularies (>50k concepts, up to millions)
- ✅ Large-scale processing (thousands of documents)
- ✅ Need guaranteed stability (no memory issues)
- ✅ Multiple entity types with different taxonomies (e.g., cancer domain)
- ✅ **Recommended for all production deployments**

### Migrating from Gazetteer to FTS5

1. Build FTS5 index from your taxonomy TSV:
   ```bash
   python src/build_fts5_indices.py \
       --taxonomy taxonomies/energy/IRENA.tsv \
       --output indices/energy/irena.db \
       --source IRENA
   ```

2. Update `domain_models.py`:
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

## 3. SemanticLinker (Linking Only)

### Description

Uses basic sentence embeddings to compute semantic similarity between entity text and taxonomy concepts.

### How It Works

```
1. Build Taxonomy Index:
   IRENA:230000 "Wind energy" → embedding_1 [768 dims]
   IRENA:240110 "Solar cell" → embedding_2 [768 dims]
   ...

2. For Each Entity:
   Query: "wind turbines" + context
   → embedding_query [768 dims]
   
3. Compute Similarities:
   similarity = cosine(embedding_query, embedding_i)
   
4. Select Best Match:
   if max(similarity) >= threshold:
       link to best match
```

### Configuration

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type semantic \
    --el_model_name intfloat/multilingual-e5-base \
    --threshold 0.6 \
    --context_window 3
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `el_model_name` | `multilingual-e5-base` | Embedding model |
| `threshold` | 0.6 | Minimum similarity for linking |
| `context_window` | 3 | Tokens around entity for context |
| `use_sentence_context` | False | Use full sentences instead |

### Model Options

```python
# Fast, smaller model (~500 MB)
"intfloat/multilingual-e5-base"

# Slower, better accuracy (~2 GB)
"intfloat/multilingual-e5-large"

# Domain-specific (if available)
"allenai/scibert_scivocab_uncased"
```

### Pros & Cons

**Pros:**
- ✅ Fast (~10-20ms per entity with cache)
- ✅ Handles paraphrases and synonyms
- ✅ No LLM required
- ✅ Works on CPU

**Cons:**
- ❌ Can produce false positives
- ❌ No explicit reasoning
- ❌ Context helps but not always sufficient
- ❌ Threshold tuning needed

### When to Use

- ✅ Processing >10k documents
- ✅ CPU-only environment
- ✅ Simple matching sufficient
- ✅ Speed > perfect accuracy

---

## 4. InstructLinker (Linking Only)

### Description

Uses instruction-tuned embeddings that better understand retrieval tasks and context.

### How It Works

Same as Semantic Linker, but uses instruction-tuned models:

```
Query format:
  "query: Given a document with <entity> in context, retrieve..."
  
Passage format:
  "passage: <taxonomy_concept_description>"
```

The model is trained to understand these instruction prefixes, leading to better semantic matching.

### Configuration

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type instruct \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.7 \
    --context_window 3
```

### Model Options

```python
# Recommended
"intfloat/multilingual-e5-large-instruct"  # Best balance

# Alternative
"intfloat/multilingual-e5-base-instruct"   # Faster, smaller
```

### Pros & Cons

**Pros:**
- ✅ Better than semantic (~5-10% improvement)
- ✅ Still reasonably fast (~15-30ms per entity)
- ✅ Better context understanding
- ✅ No LLM required

**Cons:**
- ❌ Still can produce false positives
- ❌ Larger model size than base
- ❌ Requires specific model architecture
- ❌ No explicit rejection capability

### When to Use

- ✅ Need better accuracy than semantic
- ✅ Can't use LLM (compute constraints)
- ✅ Processing medium-scale datasets
- ✅ Good balance of speed/accuracy desired

---

## 5. RerankerLinker ⭐ (Linking Only - Best Accuracy)

### Description

**Two-stage approach**: Fast embedding retrieval followed by LLM-based reranking. This is the **recommended approach** for highest-quality entity linking when accuracy is critical.

### How It Works

```
┌─────────────────────────────────────────────────┐
│ Stage 1: Embedding Retrieval (~10-20ms)         │
├─────────────────────────────────────────────────┤
│ Query: "wind turbines" (+ context?)             │
│    ↓                                            │
│ Embedding similarity search                     │
│    ↓                                            │
│ Top-k candidates:                               │
│   1. Wind energy (0.87)                         │
│   2. Wind power (0.85)                          │
│   3. Renewable energy (0.72)                    │
│   4. Energy technology (0.68)                   │
│   5. Power generation (0.65)                    │
│    ↓                                            │
│ + Optional top-level fallbacks                  │
└─────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Stage 2: LLM Reranking (~50-100ms)              │
├─────────────────────────────────────────────────┤
│ LLM Prompt:                                     │
│   Entity: "wind turbines"                       │
│   Context: "Wind turbines convert..."           │
│   Candidates: [list of 5 with details]          │
│    ↓                                            │
│ LLM evaluates with domain knowledge             │
│    ↓                                            │
│ Answer: "1" or "REJECT"                         │
└─────────────────────────────────────────────────┘
```

### Configuration

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --reranker_llm Qwen/Qwen3-1.7B \
    --threshold 0.7 \
    --context_window 3 \
    --use_context_for_retrieval false \  # IMPORTANT
    --reranker_top_k 5 \
    --reranker_fallbacks
```

### Key Parameters

#### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reranker_llm` | `Qwen/Qwen3-1.7B` | LLM for reranking |
| `reranker_top_k` | 5 | Candidates for LLM |
| `threshold` | 0.7 | Min embedding similarity |

#### Context Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_context_for_retrieval` | False | Use context in Stage 1 |
| `context_window` | 3 | Context size (tokens) |
| `max_contexts` | 3 | Max contexts per entity |
| `use_sentence_context` | False | Full sentences vs windows |

#### Advanced Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reranker_fallbacks` | True | Add top-level categories |
| `reranker_thinking` | False | Enable CoT reasoning (slower) |

### Critical: Context Usage

**Context in Stage 1 vs Stage 2:**

```python
# Stage 1 (Embedding Retrieval)
use_context_for_retrieval = False  # Default, RECOMMENDED
# → Query: "wind turbines"
# → Safer, prevents context contamination

use_context_for_retrieval = True
# → Query: "wind turbines convert kinetic energy..."
# → Riskier, but can help with ambiguous entities

# Stage 2 (LLM Reranking)
# ALWAYS uses context regardless of the above setting
# → LLM sees: entity + contexts + candidates
```

**Why separate them?**

Example problem with context in Stage 1:

```
Entity: "emissions"
Context: "reducing emissions from renewable energy systems"
         ↓
With use_context_for_retrieval=True:
  Stage 1 retrieves: "energy storage", "renewable energy"
  Stage 2 sees these candidates and might incorrectly select one
         ↓
With use_context_for_retrieval=False:
  Stage 1 retrieves: "pollution", "carbon emissions" (generic)
  Stage 2 correctly REJECTS all candidates (not energy concepts)
```

**Recommendation**: Keep `use_context_for_retrieval=False` unless you specifically need it for ambiguous entity resolution.

### LLM Models

Tested models:

```python
# Recommended (best balance)
"Qwen/Qwen3-1.7B"        # Fast, accurate, ~3.4 GB

# Alternatives
"Qwen/Qwen2.5-7B"        # More accurate, slower, ~14 GB
"Qwen/Qwen2.5-3B"        # Balanced, ~6 GB
"meta-llama/Llama-3.2-3B"  # Good alternative
```

### LLM Prompt Structure

```python
You are an {domain} domain expert. Given an entity and its context,
select the best matching concept or REJECT if none fit.

Entity: "{entity_text}"
Context: "{context}"

Candidates:
1. {label_1} ({taxonomy_id_1})
   Category: {category_1}
   Description: {description_1}
   
2. {label_2} ({taxonomy_id_2})
   Category: {category_2}
   Description: {description_2}
   
...

Instructions:
- Focus on the entity text, using context to disambiguate
- REJECT if the entity is not a true {domain} concept
- REJECT if it's a chemical compound, pollutant, or generic term
- Prefer specific matches over broad categories
- Trust the entity text over loose contextual associations

Answer with ONLY a number (1-{k}) or "REJECT".
```

### Pros & Cons

**Pros:**
- ✅✅✅ Best accuracy (~5-10% better than instruct)
- ✅ Can explicitly REJECT non-domain entities
- ✅ Handles ambiguous cases well
- ✅ Understands hierarchies (prefers specific over broad)
- ✅ Context-aware reasoning
- ✅ Domain-agnostic (works for any taxonomy)

**Cons:**
- ❌ Slower (~50-100ms per entity)
- ❌ Requires LLM (GPU recommended)
- ❌ Higher memory usage (~5-8 GB)
- ❌ More complex setup

### When to Use

- ✅ High accuracy is critical
- ✅ Have GPU available
- ✅ Processing <50k documents (reasonable speed)
- ✅ Want explicit control over false positives
- ✅ Dealing with ambiguous or challenging entities

---

## Comparison Example

Consider the entity **"emissions"** in context:

> "Renewable energy systems help reduce emissions from fossil fuels."

### Results by Linker:

| Linker | Result | Explanation |
|--------|--------|-------------|
| **Gazetteer** | No link | "emissions" not in taxonomy |
| **FTS5** | No link | "emissions" not in taxonomy |
| **Semantic** | Links to "Energy storage" | False positive (context contamination) |
| **Instruct** | Links to "Renewable energy" | False positive (better but still wrong) |
| **Reranker** | **REJECTS** | Correctly identifies as pollution, not energy concept |

---

## Performance Tuning

### Threshold Tuning

**Lower threshold → Higher recall, lower precision:**

```bash
--threshold 0.5  # Links more entities, more false positives
```

**Higher threshold → Lower recall, higher precision:**

```bash
--threshold 0.8  # Links fewer entities, fewer false positives
```

**Recommended starting points:**

| Linker | Threshold | Notes |
|--------|-----------|-------|
| Semantic | 0.6 | Start here, adjust based on results |
| Instruct | 0.7 | Can go lower (model is better) |
| Reranker | 0.7 | LLM provides additional safety |

### Context Tuning

**For better accuracy:**

```bash
--context_window 5 \
--max_contexts 5
```

**For speed:**

```bash
--context_window 2 \
--max_contexts 2  # Fewer contexts = faster
```

**For safety (Reranker only):**

```bash
--use_context_for_retrieval false  # Entity-only retrieval
```

### Cache Optimization

The cache grows over time and improves performance:

```
Cache Hit Rate:
  0-100 docs:   ~10% (cold cache)
  100-500 docs: ~40% (warming up)
  500+ docs:    ~70-80% (warm cache)
```

**Tips:**
- Process similar documents together (cache benefits)
- Preserve cache file between runs
- Monitor cache stats in logs

---

## Troubleshooting

### Problem: Low Linking Rate

**Possible causes:**
- Threshold too high
- Entities not in taxonomy
- Poor context extraction

**Solutions:**
```bash
# Lower threshold
--threshold 0.6

# Increase context
--context_window 5

# Switch to reranker for better accuracy
--linker_type reranker
```

### Problem: Too Many False Positives

**Possible causes:**
- Threshold too low
- Context contamination (Reranker)
- Taxonomy too broad

**Solutions:**
```bash
# Raise threshold
--threshold 0.8

# Disable context in Stage 1 (Reranker)
--use_context_for_retrieval false

# Reduce candidates (Reranker)
--reranker_top_k 3
```

### Problem: Slow Processing

**Possible causes:**
- LLM reranking overhead
- Cold cache
- Large context windows

**Solutions:**
```bash
# Switch to faster linker
--linker_type instruct

# Reduce context
--context_window 2

# Smaller LLM (Reranker)
--reranker_llm Qwen/Qwen3-1.7B

# Or use entity-only retrieval (Reranker)
--use_context_for_retrieval false
```

### Problem: Out of Memory

**Possible causes:**
- Large LLM model
- Too many GPU models loaded simultaneously
- Large in-memory gazetteer (FlashText)

**Solutions:**
```bash
# Use smaller LLM
--reranker_llm Qwen/Qwen3-1.7B  # ~3.4 GB

# Switch from Gazetteer to FTS5 (disk-based)
# Update domain_models.py to use linking_strategy: "fts5"

# Reduce batch size (if processing in batches)
```

### Problem: Segmentation Faults During Large-Scale Processing

**Possible causes:**
- Pandas C parser buffer issues with long text fields
- Memory fragmentation from FlashText + pandas combination

**Solutions:**
1. Switch to FTS5 linker (recommended)
2. Use Python parser for pandas: `engine='python'`
3. Process in smaller batches with explicit garbage collection

---

## Best Practices

### 1. Start Simple, Iterate

```bash
# Step 1: Try semantic first
--linker_type semantic --threshold 0.6

# Step 2: Evaluate on sample (100-200 entities)

# Step 3: If accuracy insufficient, try instruct
--linker_type instruct --threshold 0.7

# Step 4: If still insufficient, use reranker
--linker_type reranker --use_context_for_retrieval false
```

### 2. Use FTS5 for Production Exact Matching

For any domain with exact-match needs, prefer FTS5 over Gazetteer:

```python
# Instead of:
"gazetteer": {"enabled": True, "taxonomy_path": "..."}

# Use:
"gazetteer": {"enabled": False},
"linking_strategy": "fts5",
"fts5_linkers": {
    "entity_type": {
        "index_path": "indices/domain/taxonomy.db",
        "taxonomy_source": "SOURCE_NAME"
    }
}
```

### 3. Monitor Cache Performance

Check logs for cache statistics:

```
📊 Cache stats: 5234 total, 4456 linked (85.1%), 778 rejected (14.9%)
```

Good cache performance = consistent linking decisions.

### 4. Validate on Held-Out Set

Don't tune on the same data you evaluate:

```
1. Split: 80% development, 20% test
2. Tune parameters on development set
3. Final evaluation on test set only once
```

### 5. Use Reranker for Final Production (When Accuracy Matters)

Once you've established baseline with faster methods:

```bash
# Development: Fast iteration
--linker_type semantic

# Production: Best quality
--linker_type reranker --use_context_for_retrieval false
```

### 6. Document Your Configuration

Save your optimal configuration:

```bash
# energy_production_config.sh
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type reranker \
    --threshold 0.75 \
    --context_window 3 \
    --use_context_for_retrieval false \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 5 \
    --reranker_fallbacks \
    --output outputs/energy_production
```

---

## Domain-Specific Configurations

### Non-Cancer Domains (Energy, Neuro, CCAM, Maritime)

**Architecture:**
```
NER Step: GazetteerLinker (extraction + linking) + Neural NER
EL Step: SemanticLinker or RerankerLinker (linking unlinked entities)
```

**Configuration:**
```python
"energy": {
    "gazetteer": {
        "enabled": True,  # Extraction during NER
        "taxonomy_path": "taxonomies/energy/IRENA.tsv",
    },
    # EL step uses semantic/reranker (via CLI flags)
}
```

### Cancer Domain

**Architecture:**
```
NER Step: Neural NER only (AIOner) - no Gazetteer
EL Step: FTS5Linker (per entity type)
```

**Why different?**
- Large vocabularies (millions of entries)
- High ambiguity (gene symbols like "MET", "ALL", "CAT")
- Need specialized NER for biomedical text
- Scanning text with Gazetteer would produce too many false positives

**Configuration:**
```python
"cancer": {
    "gazetteer": {"enabled": False},
    "linking_strategy": "fts5",
    "fts5_linkers": {
        "gene": {"index_path": "indices/cancer/ncbi_gene.db", ...},
        "disease": {"index_path": "indices/cancer/doid_disease.db", ...},
        ...
    }
}
```

---

## Summary

| When you need... | Use this linker | Stage | Configuration |
|------------------|-----------------|-------|---------------|
| **Taxonomy-driven entity discovery** | GazetteerLinker | NER | Enable in domain config |
| **Exact matching (large vocab)** | FTS5Linker ⭐ | EL | `linking_strategy: "fts5"` |
| **Maximum speed** | SemanticLinker | EL | `--linker_type semantic --threshold 0.6` |
| **Good balance** | InstructLinker | EL | `--linker_type instruct --threshold 0.7` |
| **Best accuracy** | RerankerLinker ⭐ | EL | `--linker_type reranker --use_context_for_retrieval false` |
| **Explicit rejection** | RerankerLinker | EL | Same as above + monitor REJECT rate |
| **Large ambiguous vocabularies** | NER + FTS5 | Both | Cancer-style architecture |

**Default recommendations:**
- **Non-cancer domains**: GazetteerLinker (NER) + RerankerLinker (EL)
- **Cancer domain**: Neural NER + FTS5Linker (EL)
- **For semantic matching**: Start with **SemanticLinker** for exploration, move to **RerankerLinker** for production
