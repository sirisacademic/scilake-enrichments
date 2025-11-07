# Entity Linking - Detailed Guide

Comprehensive guide to all entity linking approaches in the SciLake pipeline.

---

## Overview

Entity Linking (NEL) maps detected entities to concepts in a controlled vocabulary (taxonomy). The pipeline offers **four linking strategies**, each with different trade-offs between speed, accuracy, and complexity.

---

## Quick Comparison

| Linker | Speed | Accuracy | Use Case | GPU |
|--------|-------|----------|----------|-----|
| **Gazetteer** | ⚡⚡⚡ Instant | 🎯🎯🎯 100% precision | Exact matches only | No |
| **Semantic** | ⚡⚡ Fast | 🎯🎯 Good | Large-scale, CPU-only | No |
| **Instruct** | ⚡ Medium | 🎯🎯🎯 Better | Balanced speed/accuracy | Optional |
| **Reranker** | 🐢 Slower | 🎯🎯🎯🎯 Best | High accuracy needs | Yes (LLM) |

---

## 1. Gazetteer Linker

### Description

Exact string matching against taxonomy terms and their aliases. Runs automatically during NER if enabled.

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

### When to Use

- ✅ You have comprehensive taxonomy with good alias coverage
- ✅ Text uses standardized terminology
- ✅ Processing speed is critical
- ✅ You need guaranteed precision

---

## 2. Semantic Linker

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

## 3. Instruct Linker

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

## 4. Reranker Linker ⭐

### Description

**Two-stage approach**: Fast embedding retrieval followed by LLM-based reranking. This is the **recommended approach** for high-quality entity linking.

### How It Works

```
┌───────────────────────────────────────────┐
│ Stage 1: Embedding Retrieval (~10-20ms)   │
├───────────────────────────────────────────┤
│ Query: "wind turbines" (+ context?)       │
│    ↓                                      │
│ Embedding similarity search               │
│    ↓                                      │
│ Top-k candidates:                         │
│   1. Wind energy (0.87)                   │
│   2. Wind power (0.85)                    │
│   3. Renewable energy (0.72)              │
│   4. Energy technology (0.68)             │
│   5. Power generation (0.65)              │
│    ↓                                      │
│ + Optional top-level fallbacks            │
└───────────────────────────────────────────┘
                  │
                  ▼
┌───────────────────────────────────────────┐
│ Stage 2: LLM Reranking (~50-100ms)        │
├───────────────────────────────────────────┤
│ LLM Prompt:                               │
│   Entity: "wind turbines"                 │
│   Context: "Wind turbines convert..."     │
│   Candidates: [list of 5 with details]    │
│    ↓                                      │
│ LLM evaluates with domain knowledge       │
│    ↓                                      │
│ Answer: "1" or "REJECT"                   │
└───────────────────────────────────────────┘
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

**Solutions:**
```bash
# Use smaller LLM
--reranker_llm Qwen/Qwen3-1.7B  # ~3.4 GB

# Use CPU
--device cpu  # (if parameter exists)

# Reduce batch size (if processing in batches)
```

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

### 2. Monitor Cache Performance

Check logs for cache statistics:

```
📊 Cache stats: 5234 total, 4456 linked (85.1%), 778 rejected (14.9%)
```

Good cache performance = consistent linking decisions.

### 3. Validate on Held-Out Set

Don't tune on the same data you evaluate:

```
1. Split: 80% development, 20% test
2. Tune parameters on development set
3. Final evaluation on test set only once
```

### 4. Use Reranker for Final Production

Once you've established baseline with faster methods:

```bash
# Development: Fast iteration
--linker_type semantic

# Production: Best quality
--linker_type reranker --use_context_for_retrieval false
```

### 5. Document Your Configuration

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

## Summary

| When you need... | Use this linker | Configuration |
|------------------|-----------------|---------------|
| **Zero false positives** | Gazetteer | Enable in domain config |
| **Maximum speed** | Semantic | `--linker_type semantic --threshold 0.6` |
| **Good balance** | Instruct | `--linker_type instruct --threshold 0.7` |
| **Best accuracy** | Reranker | `--linker_type reranker --use_context_for_retrieval false` |
| **Explicit rejection** | Reranker | Same as above + monitor REJECT rate |

**Default recommendation**: Start with **Semantic** for exploration, move to **Reranker** for production.
