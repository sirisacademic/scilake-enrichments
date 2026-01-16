# RerankerLinker - Deep Dive

Comprehensive guide to the RerankerLinker, the most sophisticated entity linking approach in the SciLake pipeline.

---

## Overview

The RerankerLinker uses a **two-stage architecture** combining:

1. **Fast embedding-based retrieval** (~10-20ms) → Get top-k candidates
2. **LLM-based reranking** (~50-100ms) → Select best match or reject

This hybrid approach achieves the **best accuracy** while maintaining reasonable speed through candidate pruning.

**Note:** RerankerLinker is the **default linker** for non-cancer domains. Configuration is centralized in `domain_models.py` under `el_config`, so you typically don't need to specify parameters manually.

---

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                   INITIALIZATION                        │
├────────────────────────────────────────────────────────┤
│  1. Load Embedding Model (multilingual-e5-large)       │
│  2. Load LLM (Qwen/Qwen3-1.7B)                         │
│  3. Load SpaCy (en_core_web_sm)                        │
│  4. Build Taxonomy Index:                              │
│     • Load IRENA.tsv                                   │
│     • Encode all concepts + aliases                    │
│     • Store: ~9000 entries × 768 dims = ~6 MB         │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│               FOR EACH ENTITY TO LINK                   │
├────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────────────────────────────────────────┐    │
│  │     STAGE 1: EMBEDDING RETRIEVAL              │    │
│  ├───────────────────────────────────────────────┤    │
│  │  Input: Entity text (+ optional context)      │    │
│  │         "wind turbines"                        │    │
│  │         "Wind turbines convert kinetic..."     │    │
│  │                                                │    │
│  │  Process:                                      │    │
│  │    1. Build query string:                     │    │
│  │       • Entity-only: "query: wind turbines"   │    │
│  │       • With context: "query: Wind turbines..."│   │
│  │                                                │    │
│  │    2. Encode query → [768-dim embedding]      │    │
│  │                                                │    │
│  │    3. Compute similarities:                   │    │
│  │       scores = query_emb @ taxonomy_embs.T    │    │
│  │                                                │    │
│  │    4. Get top-k matches (k=7):                │    │
│  │       • Wind energy (0.87)                    │    │
│  │       • Wind power (0.85)                     │    │
│  │       • Renewable energy (0.72)               │    │
│  │       • Energy technology (0.68)              │    │
│  │       • Power generation (0.65)               │    │
│  │       • ...                                   │    │
│  │                                                │    │
│  │    5. Check threshold (0.80):                 │    │
│  │       Top score 0.87 >= 0.80 ✓                │    │
│  │                                                │    │
│  │    6. Optional: Add fallbacks                 │    │
│  │       + Top-level categories                  │    │
│  │                                                │    │
│  │  Output: List of (taxonomy_id, score) tuples │    │
│  │          ~10-20ms                             │    │
│  └───────────────────────────────────────────────┘    │
│                         │                              │
│                         ▼                              │
│  ┌───────────────────────────────────────────────┐    │
│  │        STAGE 2: LLM RERANKING                 │    │
│  ├───────────────────────────────────────────────┤    │
│  │  Input: Entity + Context + Candidates         │    │
│  │                                                │    │
│  │  Build Prompt:                                │    │
│  │    You are an energy domain expert...         │    │
│  │                                                │    │
│  │    Entity: "wind turbines"                    │    │
│  │    Context: "Wind turbines convert kinetic..."│    │
│  │                                                │    │
│  │    Candidates:                                │    │
│  │    1. Wind energy (230000)                    │    │
│  │       Category: Renewables                    │    │
│  │       Desc: Wind energy is...                 │    │
│  │                                                │    │
│  │    2. Wind power (230000)                     │    │
│  │       Category: Renewables                    │    │
│  │       Desc: Wind power refers to...           │    │
│  │    ...                                        │    │
│  │                                                │    │
│  │    Instructions:                              │    │
│  │    - Focus on entity text                     │    │
│  │    - REJECT if not truly energy concept       │    │
│  │    - Prefer specific over broad               │    │
│  │    - Trust entity over context associations   │    │
│  │                                                │    │
│  │    Answer: [1-7 or REJECT]                    │    │
│  │                                                │    │
│  │  Query LLM:                                   │    │
│  │    response = llm.generate(prompt)            │    │
│  │    → "1"                                      │    │
│  │                                                │    │
│  │  Parse Response:                              │    │
│  │    Extract number or "REJECT"                 │    │
│  │    Map to (taxonomy_id, score)                │    │
│  │                                                │    │
│  │  Output: Final link or None (if rejected)     │    │
│  │          ~50-100ms                            │    │
│  └───────────────────────────────────────────────┘    │
│                                                         │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   RETURN RESULT       │
              │  • taxonomy_id        │
              │  • label              │
              │  • score              │
              │  • method: "reranker" │
              │  • candidates_info    │
              └──────────────────────┘
```

---

## Configuration

### Using Domain Defaults (Recommended)

RerankerLinker is configured as the default for non-cancer domains via `el_config` in `domain_models.py`. You typically don't need to specify any parameters:

```bash
# Uses all settings from domain el_config
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --resume
```

### Default el_config Settings

```python
"energy": {
    "el_config": {
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

### Overriding Settings

CLI arguments override domain config when specified:

```bash
# Override threshold and top_k for this run
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --threshold 0.75 \
    --reranker_top_k 10 \
    --resume
```

### Full Parameter Reference

```python
RerankerLinker(
    taxonomy_path="taxonomies/energy/IRENA.tsv",
    domain="energy",
    
    # Embedding model for Stage 1
    embedding_model_name="intfloat/multilingual-e5-large-instruct",
    
    # LLM for Stage 2
    llm_model_name="Qwen/Qwen3-1.7B",
    
    # Similarity threshold for Stage 1
    threshold=0.80,
    
    # Context extraction
    context_window=5,               # Tokens around entity
    max_contexts=5,                 # Max contexts per entity
    use_sentence_context=False,     # Use sentences vs token windows
    
    # CRITICAL: Context usage in Stage 1
    use_context_for_retrieval=False,  # Entity-only (safer)
    
    # Candidate generation
    top_k_candidates=7,             # How many for LLM
    add_top_level_fallbacks=True,   # Add broad categories
    
    # LLM behavior
    enable_thinking=False,          # Chain-of-thought (slower)
    
    logger=logger
)
```

---

## Key Parameters

### `use_context_for_retrieval` (MOST IMPORTANT)

This parameter controls whether context is used in **Stage 1 (embedding retrieval)**. Stage 2 (LLM reranking) **always uses context** regardless of this setting.

**Setting: `False` (Default, Recommended)**

```python
use_context_for_retrieval=False

# Stage 1 Query:
"query: wind turbines"

# Retrieves candidates based on entity text only
# Safer: prevents context from biasing embedding search
```

**Example where this matters:**

```
Entity: "emissions"
Context: "reducing emissions from renewable energy systems"

With use_context_for_retrieval=False:
  Stage 1: Retrieves based on "emissions" only
           → pollution, carbon emissions, GHG
  Stage 2: LLM sees candidates + context
           → Correctly REJECTS (not energy concepts)

With use_context_for_retrieval=True:
  Stage 1: Retrieves based on "emissions from renewable energy..."
           → energy storage, renewable energy (contaminated!)
  Stage 2: LLM sees these energy-related candidates
           → Might incorrectly select one (false positive)
```

**Setting: `True` (Use with caution)**

```python
use_context_for_retrieval=True

# Stage 1 Query:
"query: Wind turbines convert kinetic energy into electricity."

# Retrieves candidates based on entity + full context
# Riskier: context can contaminate retrieval
```

**When to use `True`:**
- Entity is highly ambiguous (e.g., "cell", "plant")
- You need context to disambiguate at retrieval stage
- You've tuned prompts to handle potential contamination
- You're willing to accept slightly more false positives

**Recommendation:** Start with `False`. Only switch to `True` if you have specific ambiguous cases that need it.

---

### `threshold`

Minimum similarity score for Stage 1 retrieval.

```python
# Conservative (fewer links, higher precision)
threshold=0.85

# Balanced (default)
threshold=0.80

# Aggressive (more links, lower precision)
threshold=0.6
```

**How it works:**

```python
candidates = get_top_k_similar(entity, k=7)
# Returns: [(tid_1, 0.87), (tid_2, 0.85), (tid_3, 0.72), ...]

if candidates[0][1] < threshold:
    return None  # Don't even call LLM

# Otherwise proceed to Stage 2
```

**Effect:**
- Higher → Fewer entities reach LLM (faster, but might miss valid links)
- Lower → More entities reach LLM (slower, but higher recall)

---

### `top_k_candidates`

Number of candidates to send to LLM for reranking.

```python
# Fewer candidates (faster LLM, less context)
top_k_candidates=3

# Balanced (default)
top_k_candidates=7

# More candidates (slower, but LLM has more options)
top_k_candidates=10
```

**Trade-offs:**

| k | Speed | Prompt Length | Recall | Notes |
|---|-------|---------------|--------|-------|
| 3 | Fast | Short | Lower | Best match might not be in top-3 |
| 5 | Medium | Medium | Good | Good balance |
| 7 | Medium | Medium | Better | Default, sweet spot |
| 10 | Slower | Long | Higher | Dilutes LLM focus |

---

### `add_top_level_fallbacks`

Whether to add broad top-level taxonomy categories as fallback options.

```python
add_top_level_fallbacks=True  # Default, recommended

# Example: If top candidates are all narrow,
# also include "Renewable energy", "Energy technology", etc.
```

**Why useful:**

```
Without fallbacks:
  Entity: "clean energy source"
  Candidates: Solar PV, Wind turbines, Hydropower
  LLM: None of these are exact matches
  Result: REJECT (potentially wrong)

With fallbacks:
  Entity: "clean energy source"
  Candidates: Solar PV, Wind turbines, ..., Renewable energy
  LLM: "Renewable energy" is the appropriate match
  Result: Linked (correct)
```

**Recommendation:** Keep `True` unless taxonomy has good coverage at all levels.

---

### `enable_thinking`

Enable chain-of-thought reasoning in LLM prompt.

```python
enable_thinking=False  # Default (faster)
enable_thinking=True   # Slower but more accurate
```

**With thinking:**

```
LLM Prompt:
  ...
  Think step by step:
  1. What is the entity?
  2. What is the context suggesting?
  3. Which candidate best matches?
  4. Should we reject?
  
  <thinking>
  [LLM's reasoning here]
  </thinking>
  
  Answer: 1
```

**Trade-offs:**
- ✅ More transparent reasoning
- ✅ Slightly higher accuracy (~1-2%)
- ❌ Slower (~2x response time)
- ❌ Longer prompts (token cost)

**CLI flag:** `--reranker_thinking`

**Recommendation:** Use for debugging/analysis only. Disable for production.

---

### `context_window` & `use_sentence_context`

Control how context is extracted around entities.

**Token Window Mode** (`use_sentence_context=False`):

```python
context_window=5

# Text: "... renewable wind turbines convert kinetic energy..."
#                        ↑ entity ↑
#          ← 5 tokens               5 tokens →
# Context: "renewable wind turbines convert kinetic energy into"
```

**Sentence Mode** (`use_sentence_context=True`):

```python
use_sentence_context=True

# Text: "Renewable energy is crucial. Wind turbines convert..."
#                                      ↑ entity ↑
# Context: "Wind turbines convert kinetic energy into electricity."
#          (entire sentence)
```

---

## Prompt Engineering

### Current Prompt Template

```python
f"""You are a {domain} domain expert. Given an entity and its context,
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
- Focus primarily on the entity text itself
- Use context to disambiguate only when entity is ambiguous
- REJECT if the entity is not a true {domain} concept
- REJECT if it's a chemical compound, pollutant, or measurement
- Prefer specific matches over broad categories when entity is specific
- Trust the entity text over loose contextual associations

Answer with ONLY a number (1-{k}) or "REJECT"."""
```

### Why These Instructions?

**"Focus primarily on the entity text"**
- Prevents over-reliance on context
- Example: "emissions" should be rejected even in energy context

**"REJECT if chemical compound, pollutant, measurement"**
- Domain-specific filters
- Prevents common false positives

**"Prefer specific matches over broad"**
- Encourages precision
- Example: "solar panel" → "Solar photovoltaic", not "Renewable energy"

**"Trust entity text over context"**
- Mitigates context contamination
- Example: "carbon" in energy context → reject (it's a chemical element)

---

### Customizing Prompts

For your own domain, modify:

```python
def _build_llm_prompt(self, entity_text, contexts, candidates):
    domain_specific_instructions = {
        "energy": "REJECT if chemical compound, pollutant, or measurement",
        "biology": "REJECT if anatomical part without biological process",
        "medicine": "REJECT if symptom without disease indication",
    }
    
    instructions = domain_specific_instructions.get(self.domain, "")
    
    prompt = f"""You are a {self.domain} domain expert...
    
    {instructions}
    
    Entity: {entity_text}
    ...
    """
```

---

## Performance Optimization

### Cache Strategy

The RerankerLinker benefits hugely from caching:

```python
# First time linking "wind turbines":
1. Stage 1 retrieval: ~10ms
2. Stage 2 LLM: ~80ms
Total: ~90ms

# Second time (cache hit):
1. Check cache: ~1ms
Total: ~1ms (90x faster!)
```

**Cache growth pattern:**

```
Cache Size Over Documents:

8000 │                                    ╭────────
     │                           ╭────────╯
6000 │                    ╭──────╯
     │              ╭─────╯
4000 │         ╭────╯
     │    ╭────╯
2000 │╭───╯
     │
   0 └───┬───┬───┬───┬───┬───┬───┬────
       0  100 200 500 1k  2k  5k  10k+ docs

Hit Rate:
  First 100:  ~10%
  First 500:  ~50%
  First 1000: ~70%
  5000+:      ~80-85%
```

**Optimization tips:**

1. **Preserve cache between runs:**
   ```bash
   # Cache saved at: outputs/<domain>/el/cache/linking_cache.json
   # Keep this file!
   ```

2. **Process similar documents together:**
   ```bash
   # Bad: Process random docs from different topics
   # Good: Process all energy policy papers, then all technical papers
   ```

3. **Monitor cache stats:**
   ```python
   logger.info(f"Cache: {hits} hits, {misses} misses")
   ```

---

### Batching Strategy

Process documents in batches to manage memory:

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --batch_size 100 \
    --resume
```

**Why batch?**
- Checkpoint after each batch (safe resumption)
- Save cache periodically (avoid loss)
- Monitor progress granularly

---

### GPU Memory Management

**Typical memory usage:**

```
Component                 Memory
─────────────────────────────────
Embedding model           ~500 MB
LLM (Qwen3-1.7B)         ~3.4 GB
SpaCy                     ~200 MB
GLiNER (if running NER)  ~500 MB
RoBERTa (if running NER) ~500 MB
Working memory           ~500 MB
─────────────────────────────────
TOTAL                    ~5-6 GB
```

**If OOM (Out of Memory):**

1. **Use smaller LLM:**
   ```bash
   --reranker_llm Qwen/Qwen3-1.7B  # Smallest recommended
   ```

2. **Run NER and EL separately:**
   ```bash
   # Step 1: NER only (frees GPU after)
   --step ner
   
   # Step 2: EL only
   --step el
   ```

3. **Reduce candidates:**
   ```bash
   --reranker_top_k 3  # Shorter prompts
   ```

4. **CPU fallback (slow):**
   ```bash
   # If parameter exists:
   --device cpu
   ```

---

## Troubleshooting

### High Rejection Rate

**Symptom:** LLM rejects >30% of entities

**Possible causes:**
1. Threshold too low (too many weak candidates reach LLM)
2. Taxonomy coverage gaps
3. Prompt too strict

**Solutions:**

```bash
# Raise threshold
--threshold 0.85

# Add fallbacks (enabled by default)
--reranker_fallbacks

# Check logs for rejection patterns:
grep "REJECT" outputs/energy/logs/energy_el.log | head -20
```

---

### High False Positive Rate

**Symptom:** Incorrect links to energy concepts

**Possible causes:**
1. `use_context_for_retrieval=True` causing contamination
2. Threshold too low
3. Too many broad candidates

**Solutions:**

```bash
# Disable context in Stage 1 (already default)
--use_context_for_retrieval false

# Raise threshold
--threshold 0.85

# Reduce candidates
--reranker_top_k 3

# Update prompt with stricter instructions
```

---

### Slow Processing

**Symptom:** <10 entities/sec

**Possible causes:**
1. Cold cache
2. LLM inference slow
3. Too many candidates

**Solutions:**

```bash
# Use smaller LLM (default)
--reranker_llm Qwen/Qwen3-1.7B

# Reduce candidates
--reranker_top_k 3

# Disable thinking mode (already default)
# (Don't use --reranker_thinking)

# Reduce context
--context_window 2 --max_contexts 2

# Wait for cache to warm up (first 500 docs are slower)
```

---

### LLM Returns Invalid Answers

**Symptom:** LLM doesn't return number or "REJECT"

**Possible causes:**
1. Prompt ambiguity
2. Model not following instructions
3. Unusual entity/context

**Debugging:**

```python
# Enable debug logging
logger.setLevel(logging.DEBUG)

# Check LLM raw output
logger.debug(f"LLM answer: {answer}")
```

**Solutions:**
- Use better instruction-following models (Qwen works well)
- Simplify prompt
- Add few-shot examples

---

## Advanced Usage

### Custom Taxonomy

```python
# Your taxonomy must have columns:
# id | concept | wikidata_id | wikidata_aliases | parent_id | type | description

# Load custom taxonomy:
reranker = RerankerLinker(
    taxonomy_path="taxonomies/my_domain/my_taxonomy.tsv",
    domain="my_domain",
    ...
)
```

**Taxonomy requirements:**
- TSV format with tab separators
- Required columns: `id`, `concept`
- Optional but recommended: `wikidata_id`, `wikidata_aliases`, `description`, `type`
- Aliases: pipe-separated (e.g., "wind power|wind energy|wind turbines")

---

### Domain-Specific Prompts

Modify prompts for your domain:

```python
# In reranker_linker.py, _build_llm_prompt():

domain_instructions = {
    "energy": """
- REJECT if chemical compound, pollutant, or measurement
- REJECT if generic process not energy-specific
""",
    "biology": """
- REJECT if anatomical part without biological function
- REJECT if chemical compound without biological context
""",
    "medicine": """
- REJECT if general symptom without disease context
- Prefer specific diseases over broad categories
"""
}
```

---

### Monitoring & Evaluation

**Track key metrics:**

```python
# Add to your pipeline:
stats = {
    "total_entities": 0,
    "linked": 0,
    "rejected": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "avg_score": 0.0,
    "avg_time_ms": 0.0
}

# Log every N files:
if file_count % 100 == 0:
    logger.info(f"Stats: {stats}")
```

**Quality checks:**

```bash
# Sample 100 entities for manual review
python tools/sample_for_review.py \
    --input outputs/energy/el/ \
    --output review_sample.csv \
    --n 100
```

---

## Comparison with Other Linkers

### RerankerLinker vs InstructLinker

| Aspect | Reranker | Instruct |
|--------|----------|----------|
| Accuracy | ~93% | ~88% |
| Speed | ~70ms/entity | ~20ms/entity |
| False Positives | Lower | Higher |
| Explicit Rejection | Yes | No |
| GPU Required | Yes (LLM) | Optional |
| Best For | High quality | Speed/scale |

**When to choose Reranker:**
- ✅ Quality is critical
- ✅ Have GPU available
- ✅ Processing <50k docs
- ✅ Want explicit control

**When to choose Instruct:**
- ✅ Speed matters
- ✅ Limited GPU
- ✅ Processing >50k docs
- ✅ Good enough accuracy

---

## Summary

### Key Takeaways

1. **Two-stage architecture**: Fast retrieval + accurate reranking
2. **Context control**: Separate Stage 1 (optional) and Stage 2 (always)
3. **Safety first**: `use_context_for_retrieval=False` prevents contamination
4. **Cache is king**: Performance improves dramatically over time
5. **Prompt matters**: Domain-specific instructions reduce false positives
6. **Use defaults**: el_config provides optimized settings per domain

### Recommended Usage

**Simple (recommended):** Let el_config handle configuration:

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --resume
```

**With overrides:** When you need to tune specific parameters:

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --threshold 0.75 \
    --reranker_top_k 10 \
    --resume
```

### Default Configuration (from el_config)

| Parameter | Default Value |
|-----------|---------------|
| `linker_type` | `reranker` |
| `el_model_name` | `intfloat/multilingual-e5-large-instruct` |
| `threshold` | `0.80` |
| `context_window` | `5` |
| `max_contexts` | `5` |
| `reranker_llm` | `Qwen/Qwen3-1.7B` |
| `reranker_top_k` | `7` |
| `reranker_fallbacks` | `True` |
| `use_context_for_retrieval` | `False` |
| `enable_thinking` | `False` |

### Next Steps

1. Run on small sample (100 docs) with defaults
2. Evaluate manually (sample 50-100 entities)
3. Tune threshold if precision/recall needs adjustment
4. Scale to full dataset
5. Monitor cache performance
6. Iterate on prompts if needed
