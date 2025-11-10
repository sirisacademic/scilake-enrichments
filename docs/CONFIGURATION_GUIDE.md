# Configuration Guide

Practical configuration examples and best practices for the SciLake NER & Entity Linking pipeline.

---

## Overview

The pipeline is highly configurable through:

1. **Command-line arguments** (runtime behavior)
2. **Domain configuration files** (domain-specific models)
3. **Taxonomy files** (knowledge bases)

---

## Domain Configuration

### Location

```
configs/domain_models.py
```

### Structure

```python
DOMAIN_MODELS = {
    "energy": {
        "ner_models": {
            "gliner": {
                "enabled": True,
                "model_name": "urchade/gliner_multi",
                "labels": [
                    "energy technology",
                    "energy storage",
                    "energy type",
                    "energy source",
                    "transportation",  # Important for disambiguation!
                    "measurement unit"
                ]
            },
            "roberta": {
                "enabled": True,
                "model_name": "energy-roberta-ner",
                "labels": ["energytype", "energystorage", ...]
            }
        },
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "case_sensitive": False
        },
        "entity_linking": {
            "default_linker": "reranker",
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "taxonomy_source": "IRENA",
            "threshold": 0.7
        }
    },
    
    "neuro": {
        "ner_models": {
            "gliner": {
                "enabled": True,
                "model_name": "urchade/gliner_multi",
                "labels": [
                    "brain region",
                    "neuron type",
                    "neural pathway",
                    "neurotransmitter",
                    "disease"  # For disambiguation
                ]
            }
        },
        "entity_linking": {
            "default_linker": "semantic",
            "taxonomy_path": "taxonomies/neuro/openMINDS.tsv",
            "taxonomy_source": "openMINDS",
            "threshold": 0.6
        }
    }
}
```

### Key Design Principles

#### 1. Multi-Label NER Configuration

**Critical:** GLiNER uses semantic similarity, not exact matching. Providing multiple label options dramatically improves accuracy:

```python
# Bad (single label, ambiguous)
"labels": ["energy technology"]

# Good (multiple labels, disambiguates)
"labels": [
    "energy technology",
    "energy storage",
    "transportation",  # Helps reject cars/trains
    "measurement unit"  # Helps reject kWh/MW
]
```

**Example:**

```
Entity: "electric vehicle"

With single label:
  ["energy technology"] → Matches (WRONG, it's transportation)

With multiple labels:
  ["energy technology", "transportation"] 
  → Matches "transportation" better (CORRECT, rejects from energy)
```

#### 2. Domain-Specific Thresholds

Different domains need different thresholds:

```python
"energy": {
    "threshold": 0.7  # Energy taxonomy is well-structured
}

"neuro": {
    "threshold": 0.6  # Neuroanatomy terms more ambiguous
}

"maritime": {
    "threshold": 0.75  # Very specific technical terms
}
```

---

## Command-Line Configuration

### Complete Example

```bash
python src/pipeline.py \
    # Core settings
    --domain energy \
    --input data/energy/papers \
    --output outputs/energy_run_20251107 \
    --step all \
    --resume \
    --batch_size 100 \
    \
    # Entity Linking
    --linker_type reranker \
    --threshold 0.7 \
    --taxonomy taxonomies/energy/IRENA.tsv \
    --taxonomy_source IRENA \
    \
    # Context extraction
    --context_window 3 \
    --max_contexts 3 \
    --use_sentence_context \
    \
    # Reranker-specific
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 5 \
    --reranker_fallbacks \
    --use_context_for_retrieval false \
    \
    # Debugging
    --debug
```

---

## Configuration Recipes

### 1. High Precision (Minimize False Positives)

**Use case:** Scientific publications, regulatory documents

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy_high_precision \
    --step all \
    --linker_type reranker \
    --threshold 0.8 \                      # High threshold
    --use_context_for_retrieval false \    # Entity-only retrieval
    --reranker_top_k 3 \                   # Fewer candidates
    --reranker_fallbacks false \           # No broad categories
    --context_window 2 \                   # Less context
    --resume
```

**Expected results:**
- Precision: ~95%
- Linking rate: ~75%
- Speed: ~80ms/entity

---

### 2. High Recall (Maximum Coverage)

**Use case:** Initial exploration, broad surveys

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy_high_recall \
    --step all \
    --linker_type reranker \
    --threshold 0.6 \                      # Lower threshold
    --use_context_for_retrieval true \     # Use context for retrieval
    --reranker_top_k 10 \                  # More candidates
    --reranker_fallbacks \                 # Include broad categories
    --context_window 5 \                   # More context
    --max_contexts 5 \
    --use_sentence_context \               # Full sentences
    --resume
```

**Expected results:**
- Precision: ~85%
- Linking rate: ~90%
- Speed: ~120ms/entity

---

### 3. Maximum Speed (Large-Scale Processing)

**Use case:** 20k+ documents, tight deadlines

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy_fast \
    --step all \
    --linker_type semantic \               # Fastest linker
    --threshold 0.7 \
    --context_window 2 \                   # Minimal context
    --max_contexts 2 \
    --batch_size 500 \                     # Large batches
    --resume
```

**Expected results:**
- Precision: ~88%
- Linking rate: ~82%
- Speed: ~15ms/entity (after cache warms)

---

### 4. Balanced (Production Default)

**Use case:** General purpose, good quality

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy_production \
    --step all \
    --linker_type reranker \
    --threshold 0.7 \
    --use_context_for_retrieval false \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 5 \
    --reranker_fallbacks \
    --context_window 3 \
    --max_contexts 3 \
    --use_sentence_context \
    --batch_size 100 \
    --resume
```

**Expected results:**
- Precision: ~92%
- Linking rate: ~85%
- Speed: ~70ms/entity

---

### 5. Development/Testing (Fast Iteration)

**Use case:** Rapid experimentation, tuning

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy/sample \           # Small sample
    --output outputs/energy_dev \
    --step all \
    --linker_type semantic \               # Fast for testing
    --threshold 0.65 \
    --context_window 3 \
    --batch_size 10 \                      # Small batches
    --debug \                              # Verbose logging
    --resume
```

**Tips:**
- Use small sample first (~100 docs)
- Iterate quickly
- Switch to reranker once config is stable

---

## Taxonomy Preparation

### Required Format

```tsv
taxonomy_id	label	wikidata_id	aliases	parent_id	category	description
230000	Wind energy	Q43302	wind power|wind turbines	200000	Renewables	Wind energy is...
240110	Solar cell	Q15171558	PV cell|photovoltaic cell	240000	Renewables	A solar cell...
```

### Best Practices

#### 1. Rich Aliases

```tsv
# Bad (minimal aliases)
230000	Wind energy	Q43302	wind power

# Good (comprehensive aliases)
230000	Wind energy	Q43302	wind power|wind turbines|wind farm|wind generation|aeolian energy
```

**Why:** More aliases → better gazetteer coverage → fewer entities need semantic linking.

#### 2. Hierarchical Structure

```tsv
# Top-level
200000	Renewables		renewable energy|clean energy		Energy	Renewable energy sources

# Mid-level
230000	Wind energy	Q43302	wind power	200000	Renewables	Wind energy generation

# Specific
231000	Onshore wind	Q...	land-based wind	230000	Renewables	Wind turbines on land
231100	Offshore wind	Q...	marine wind	230000	Renewables	Wind turbines at sea
```

**Why:** Hierarchy enables:
- Top-level fallbacks in RerankerLinker
- Better LLM understanding
- Navigation of concepts

#### 3. Good Descriptions

```tsv
# Bad (tautological)
230000	Wind energy	Q43302		200000	Renewables	Energy from wind

# Good (informative)
230000	Wind energy	Q43302		200000	Renewables	Wind energy is the conversion of kinetic energy from wind into electricity using wind turbines. Wind turbines extract energy from moving air masses to generate renewable electricity.
```

**Why:** Descriptions help:
- LLM reranking
- Human reviewers
- Disambiguation

---

### Wikidata Enhancement

Use SPARQL to enrich taxonomy with Wikidata aliases:

```sparql
SELECT ?item ?label ?alias WHERE {
  VALUES ?item { wd:Q43302 }  # Wind energy
  ?item rdfs:label ?label .
  OPTIONAL { ?item skos:altLabel ?alias }
  FILTER(LANG(?label) = "en")
  FILTER(!BOUND(?alias) || LANG(?alias) = "en")
}
```

**Script example:**

```python
# tools/enrich_taxonomy.py
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

def get_wikidata_aliases(wikidata_id):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    
    query = f"""
    SELECT ?alias WHERE {{
      wd:{wikidata_id} skos:altLabel ?alias .
      FILTER(LANG(?alias) = "en")
    }}
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    return [r['alias']['value'] for r in results['results']['bindings']]

# Load taxonomy
df = pd.read_csv("taxonomies/energy/IRENA.tsv", sep="\t")

# Enrich aliases
for idx, row in df.iterrows():
    if pd.notna(row['wikidata_id']):
        wd_id = row['wikidata_id'].split('/')[-1]
        aliases = get_wikidata_aliases(wd_id)
        
        # Merge with existing aliases
        existing = row['aliases'].split('|') if pd.notna(row['aliases']) else []
        all_aliases = list(set(existing + aliases))
        df.at[idx, 'aliases'] = '|'.join(all_aliases)

# Save enriched taxonomy
df.to_csv("taxonomies/energy/IRENA_enriched.tsv", sep="\t", index=False)
```

---

## Environment Setup

### Conda Environment

```yaml
# environment.yml
name: scilake-enrichments
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch>=2.0.0
      - transformers>=4.35.0
      - sentence-transformers>=2.2.0
      - spacy>=3.5.0
      - pandas>=2.0.0
      - numpy>=1.24.0
      - tqdm>=4.65.0
      - flashtext>=2.7
      - rdflib>=6.3.0
      - abbreviations>=0.2.0
```

### Hardware Recommendations

| Configuration | CPU | RAM | GPU | Use Case |
|---------------|-----|-----|-----|----------|
| **Minimum** | 4 cores | 8 GB | None | Semantic linker only, small datasets |
| **Recommended** | 8 cores | 16 GB | 8 GB VRAM | Reranker, medium datasets |
| **Optimal** | 16+ cores | 32 GB | 16 GB VRAM | Full pipeline, large datasets |

**Notes:**
- Semantic/Instruct linkers can run on CPU (slower)
- RerankerLinker requires GPU for reasonable speed
- More RAM → larger batches → better throughput

---

## Logging Configuration

### Log Levels

```python
# In pipeline.py
logger = setup_logger(
    log_dir="outputs/energy/logs",
    name="energy_pipeline",
    debug=True  # Set via --debug flag
)
```

**Debug mode (`--debug`):**
- Logs every entity linking decision
- Shows LLM prompts and responses
- Tracks cache hits/misses
- **Use for:** Development, troubleshooting

**Info mode (default):**
- Logs file-level progress
- Shows summary statistics
- Periodic cache stats
- **Use for:** Production

### Log Analysis

**Find rejection patterns:**

```bash
grep "REJECT" outputs/energy/logs/energy_el.log \
  | cut -d"'" -f2 \
  | sort \
  | uniq -c \
  | sort -rn \
  | head -20
```

**Track linking rate over time:**

```bash
grep "entities linked" outputs/energy/logs/energy_el.log \
  | awk '{print $6}' \
  | cut -d'/' -f1
```

**Monitor cache performance:**

```bash
grep "Cache:" outputs/energy/logs/energy_el.log \
  | tail -20
```

---

## Checkpoint Management

### Checkpoint Structure

```
outputs/
└── energy/
    ├── checkpoints/
    │   └── processed.json     # Tracks completed files
    ├── el/
    │   └── cache/
    │       └── linking_cache.json  # Entity linking cache
    └── logs/
```

### Resume from Checkpoint

```bash
# Interrupted run
python src/pipeline.py --domain energy --step all --output outputs/energy

# Resume (automatically skips processed files)
python src/pipeline.py --domain energy --step all --output outputs/energy --resume
```

### Manual Checkpoint Inspection

```python
import json

# Check processed files
with open("outputs/energy/checkpoints/processed.json") as f:
    processed = json.load(f)
    print(f"Processed {len(processed)} files")
    print("Last file:", list(processed.keys())[-1])

# Check cache size
with open("outputs/energy/el/cache/linking_cache.json") as f:
    cache = json.load(f)
    linked = sum(1 for v in cache.values() if v is not None)
    rejected = sum(1 for v in cache.values() if v is None)
    print(f"Cache: {linked} linked, {rejected} rejected")
```

---

## Testing Configuration

### Unit Tests

```bash
# Test individual components
python -m pytest tests/test_nif_reader.py
python -m pytest tests/test_semantic_linker.py
python -m pytest tests/test_reranker_linker.py
```

### Integration Tests

```bash
# Test full pipeline on small sample
python src/pipeline.py \
    --domain energy \
    --input tests/data/sample \
    --output tests/output \
    --step all \
    --batch_size 5
```

### Validation Script

```python
# tools/validate_output.py
import json
import pandas as pd

def validate_linking(jsonl_file, min_linking_rate=0.8):
    """Check linking quality"""
    entities = []
    
    with open(jsonl_file) as f:
        for line in f:
            doc = json.loads(line)
            entities.extend(doc['entities'])
    
    total = len(entities)
    linked = sum(1 for e in entities if e.get('linking'))
    
    linking_rate = linked / total if total > 0 else 0
    
    print(f"Total entities: {total}")
    print(f"Linked: {linked} ({linking_rate:.1%})")
    
    if linking_rate < min_linking_rate:
        print(f"WARNING: Linking rate below {min_linking_rate:.1%}")
        return False
    
    return True

# Usage
validate_linking("outputs/energy/el/paper1.jsonl")
```

---

## Troubleshooting Common Issues

### Issue: Low Linking Rate

**Check:**
1. Threshold too high?
2. Taxonomy coverage?
3. Entity detection quality?

**Debug:**

```bash
# Run with debug logging
--debug

# Check entity types being detected
grep "entity" outputs/energy/logs/energy_ner.log | head -50

# Check why entities aren't linking
grep "below threshold" outputs/energy/logs/energy_el.log | head -20
```

---

### Issue: Too Many False Positives

**Check:**
1. `use_context_for_retrieval=true` (change to false)
2. Threshold too low
3. GLiNER labels too broad

**Debug:**

```bash
# Sample false positives
python tools/sample_for_review.py \
    --input outputs/energy/el \
    --filter_by high_score \
    --output review_fps.csv
```

---

### Issue: Cache Not Working

**Check:**
1. Cache file exists and is writable
2. Cache format valid JSON
3. No permission issues

**Fix:**

```bash
# Reset cache
rm outputs/energy/el/cache/linking_cache.json

# Verify permissions
chmod 644 outputs/energy/el/cache/linking_cache.json
```

---

## Summary

### Quick Start Checklist

- [ ] Set up environment (`conda env create -f environment.yml`)
- [ ] Configure domain in `configs/domain_models.py`
- [ ] Prepare taxonomy (required columns + good aliases)
- [ ] Choose linker type (semantic → instruct → reranker)
- [ ] Run on small sample first
- [ ] Tune threshold based on precision/recall needs
- [ ] Scale to full dataset with `--resume`

### Best Practices

1. **Start simple:** Use semantic linker first
2. **Test small:** Run on 100 docs before scaling
3. **Monitor logs:** Check linking rate and cache performance
4. **Preserve cache:** Keep `linking_cache.json` between runs
5. **Use checkpoints:** Always enable `--resume`
6. **Validate regularly:** Sample for manual review

### Configuration Templates

Save these for reuse:

```bash
# configs/energy_production.sh
python src/pipeline.py \
    --domain energy \
    --step all \
    --linker_type reranker \
    --threshold 0.7 \
    --use_context_for_retrieval false \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 5 \
    --context_window 3 \
    --use_sentence_context \
    --batch_size 100 \
    --resume
```
