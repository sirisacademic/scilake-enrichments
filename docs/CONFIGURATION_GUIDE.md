# SciLake Pipeline Configuration Guide

This guide provides configuration recipes for different use cases, including entity filtering, parallel processing, and performance optimization.

---

## Table of Contents

1. [Domain Configuration](#domain-configuration)
2. [Entity Linking Configuration (el_config)](#entity-linking-configuration-el_config)
3. [Entity Filtering](#entity-filtering)
4. [Taxonomy Preparation](#taxonomy-preparation)
5. [Environment Setup](#environment-setup)
6. [Input Format Configuration](#input-format-configuration)
7. [Parallel Processing](#parallel-processing)
8. [Performance Optimization](#performance-optimization)
9. [Configuration Recipes](#configuration-recipes)
10. [Checkpoint Management](#checkpoint-management)
11. [Logging Configuration](#logging-configuration)
12. [Testing Configuration](#testing-configuration)
13. [Troubleshooting Common Issues](#troubleshooting-common-issues)

---

## Domain Configuration

### Basic Domain Setup

In `src/domain_models.py`:

```python
DOMAIN_MODELS = {
    "energy": {
        # NER Models
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Energy-roberta-base",
                "type": "roberta",
                "entity_types": ["energytype"],
            }
        ],
        
        # Gazetteer (extraction + linking)
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "taxonomy_source": "IRENA",
            "model_name": "IRENA-Gazetteer",
            "default_type": "energytype",
            "min_term_length": 2,
        },
        
        # Entity Filtering
        "min_mention_length": 2,
        "blocked_mentions": {"energy", "power", "system"},
        
        # Entity Linking Configuration (see el_config section)
        "el_config": {...},
        
        # Type Matching
        "enforce_type_match": True,
        "type_mappings": {...},
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

**Why this works:**
- GLiNER calculates semantic similarity between entity context and label
- Multiple labels give the model alternative categories
- The model picks the highest-scoring label
- This prevents misclassification of ambiguous entities

#### 2. Domain-Specific Thresholds

Different domains need different thresholds based on taxonomy structure. These are now configured in `el_config`:

```python
"energy": {
    "el_config": {
        "threshold": 0.80  # Energy taxonomy is well-structured
    }
}

"neuro": {
    "el_config": {
        "threshold": 0.80  # Can be lowered if linking rate is too low
    }
}
```

### Adding a New Domain

1. **Create taxonomy file:**
   ```
   taxonomies/newdomain/taxonomy.tsv
   ```
   
   Required columns: `id`, `concept`, `type`
   Optional columns: `description`, `synonyms`, `wikidata_id`, `wikidata_aliases`

2. **Add domain configuration:**
   ```python
   "newdomain": {
       "models": [...],
       "gazetteer": {...},
       "min_mention_length": 2,
       "blocked_mentions": set(),
       "el_config": {
           "taxonomy_path": "taxonomies/newdomain/taxonomy.tsv",
           "taxonomy_source": "NewDomain",
           "linker_type": "reranker",
           "threshold": 0.80,
           ...
       },
   }
   ```

3. **Test on sample:**
   ```bash
   python src/pipeline.py --domain newdomain --input data/sample --output outputs/test --step all --resume
   ```

---

## Entity Linking Configuration (el_config)

Entity linking parameters are centralized in the `el_config` section of each domain in `domain_models.py`. This simplifies running the pipeline - you typically only need to specify domain and paths.

### el_config Structure

```python
"energy": {
    # ... NER config ...
    
    "el_config": {
        # Taxonomy settings
        "taxonomy_path": "taxonomies/energy/IRENA.tsv",
        "taxonomy_source": "IRENA",
        
        # Linker type
        "linker_type": "reranker",  # "semantic" | "instruct" | "reranker" | "fts5"
        
        # Embedding model
        "el_model_name": "intfloat/multilingual-e5-large-instruct",
        
        # Threshold and context
        "threshold": 0.80,
        "context_window": 5,
        "max_contexts": 5,
        "use_sentence_context": False,
        
        # Reranker-specific settings
        "reranker_llm": "Qwen/Qwen3-1.7B",
        "reranker_top_k": 7,
        "reranker_fallbacks": True,
    },
}
```

### Default Values

| Parameter | Default | Description |
|-----------|---------|-------------|
| `linker_type` | `"reranker"` | Linking strategy |
| `el_model_name` | `"intfloat/multilingual-e5-large-instruct"` | Embedding model |
| `threshold` | `0.80` | Similarity threshold |
| `context_window` | `5` | Token context window |
| `max_contexts` | `5` | Max contexts per entity |
| `use_sentence_context` | `False` | Use full sentences |
| `reranker_llm` | `"Qwen/Qwen3-1.7B"` | LLM for reranking |
| `reranker_top_k` | `7` | Candidates for reranker |
| `reranker_fallbacks` | `True` | Add top-level fallbacks |

### CLI Override

CLI arguments override domain config when specified:

```bash
# Uses all settings from el_config
python src/pipeline.py --domain energy --output outputs/energy --step el --resume

# Override threshold for this run only
python src/pipeline.py --domain energy --output outputs/energy --step el --threshold 0.75 --resume

# Override multiple settings
python src/pipeline.py --domain energy --output outputs/energy --step el \
    --threshold 0.70 \
    --reranker_top_k 10 \
    --resume
```

### Parameter Resolution Order

1. CLI argument (if specified)
2. Domain `el_config` (if defined)
3. Hardcoded fallback default

### Cancer Domain (FTS5)

Cancer domain uses FTS5 linking instead of reranker:

```python
"cancer": {
    "gazetteer": {"enabled": False},
    "linking_strategy": "fts5",
    
    "fts5_linkers": {
        "gene": {"index_path": "indices/cancer/ncbi_gene.db", "taxonomy_source": "NCBI_Gene"},
        "disease": {"index_path": "indices/cancer/doid_disease.db", "taxonomy_source": "DOID"},
        # ...
    },
    
    "el_config": {
        "linker_type": "fts5",
        # Minimal config - FTS5 uses exact matching
    },
    
    # Type matching is implicit via FTS5 routing (each index is type-specific)
    "enforce_type_match": False,
}
```

---

## Entity Filtering

Entity filtering helps reduce false positives by blocking generic terms and very short mentions that are likely noise.

### Minimum Mention Length

Skip entities shorter than a specified character count.

**Global setting (all entity types):**
```python
"min_mention_length": 2,
```

**Per-entity-type settings:**
```python
"min_mention_length": {
    "gene": 2,      # Gene symbols can be short (TP53)
    "disease": 3,   # Disease names usually longer
    "species": 4,   # Species names longer
    "_default": 2,  # Fallback for unspecified types
},
```

### Blocked Mentions

Skip specific terms that are too generic or cause false positives.

**Global blocked list (all entity types):**
```python
"blocked_mentions": {"energy", "power", "data", "system", "model", "method"},
```

**Per-entity-type blocked lists:**
```python
"blocked_mentions": {
    "species": {"patient", "patients", "man", "woman", "human", "people"},
    "disease": {"pain", "syndrome", "condition", "disorder"},
    "gene": {"gene", "protein", "factor"},
},
```

### Complete Filtering Example

```python
"cancer": {
    "models": [...],
    
    "gazetteer": {"enabled": False},  # Cancer uses FTS5
    
    "linking_strategy": "fts5",
    
    "fts5_linkers": {
        "gene": {
            "index_path": "indices/cancer/ncbi_gene.db",
            "taxonomy_source": "NCBI_Gene",
        },
        "disease": {
            "index_path": "indices/cancer/doid_disease.db",
            "taxonomy_source": "DOID",
        },
        "species": {
            "index_path": "indices/cancer/ncbi_species.db",
            "taxonomy_source": "NCBI_Taxonomy",
        },
    },
    
    # Entity Filtering
    "min_mention_length": {
        "gene": 2,
        "disease": 3,
        "species": 4,
        "cellline": 3,
        "_default": 2,
    },
    
    "blocked_mentions": {
        "species": {
            "patient", "patients", "man", "men", "woman", "women",
            "human", "humans", "people", "person", "individual",
            "child", "children", "adult", "adults", "infant", "infants",
        },
        "disease": {
            "pain", "syndrome", "disease", "disorder", "condition",
            "symptom", "symptoms", "sign", "signs",
        },
        "gene": {
            "gene", "genes", "protein", "proteins",
        },
    },
}
```

---

## Taxonomy Preparation

A well-prepared taxonomy is critical for high-quality entity linking.

### Required Format

```tsv
id	concept	wikidata_id	synonyms	parent_id	type	description
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

## Input Format Configuration

### NIF Format (Default)

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --input_format nif \
    --step all \
    --resume
```

### Title/Abstract JSON Format

**Input file structure:**
```json
{"oaireid": "50|doi_dedup___::abc", "titles": ["Title"], "abstracts": ["Abstract..."]}
{"oaireid": "50|doi_dedup___::def", "titles": ["Title 2"], "abstracts": ["Abstract 2..."]}
```

**Command:**
```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy_titleabstract.json \
    --output outputs/energy-ta \
    --input_format title_abstract \
    --step all \
    --resume
```

### Legal Text JSON Format

**Input file structure:**
```json
{"rsNr": "0.101", "en_lawTitle": "Convention...", "en_lawText": "Full text..."}
{"rsNr": "0.102", "en_lawTitle": "Regulation...", "en_lawText": "Full text..."}
```

**Command:**
```bash
python src/pipeline.py \
    --domain energy \
    --input data/fedlex-dataset.jsonl \
    --output outputs/energy-legal \
    --input_format legal_text \
    --step all \
    --resume
```

---

## Parallel Processing

Parallel processing is recommended for large datasets (>10K documents) to maximize throughput.

### When to Use Parallel Processing

| Dataset Size | Recommendation |
|--------------|----------------|
| <10K sections | Single process |
| 10K-100K sections | 2-3 parallel processes |
| 100K-1M sections | 4-6 parallel processes |
| >1M sections | 6-8 parallel processes |

### Step 1: Split Input File

```bash
# Split into N parts (e.g., 6)
split -n l/6 -d --additional-suffix=.json \
    data/energy_titleabstract.json \
    data/energy_titleabstract_part
```

This creates:
- `energy_titleabstract_part00.json`
- `energy_titleabstract_part01.json`
- ...
- `energy_titleabstract_part05.json`

### Step 2: Run Parallel NER

```bash
#!/bin/bash
# run_ner_parallel.sh

for i in 00 01 02 03 04 05; do
    echo "Starting NER part ${i}..."
    nohup python src/pipeline.py \
        --domain energy \
        --step ner \
        --input_format title_abstract \
        --input data/energy_titleabstract_part${i}.json \
        --output outputs/energy-part${i} \
        --resume \
        > outputs/energy-part${i}_ner.log 2>&1 &
done

echo "All NER instances started. Monitor with: tail -f outputs/energy-part*_ner.log"
```

### Step 3: Wait for NER Completion

```bash
# Check progress
tail -f outputs/energy-part00_ner.log

# Check if all processes are done
ps aux | grep pipeline.py
```

### Step 4: Run Parallel Entity Linking

EL configuration is automatically loaded from domain el_config:

```bash
#!/bin/bash
# run_el_parallel.sh

for i in 00 01 02 03 04 05; do
    echo "Starting EL part ${i}..."
    nohup python src/pipeline.py \
        --domain energy \
        --step el \
        --output outputs/energy-part${i} \
        --resume \
        > outputs/energy-part${i}_el.log 2>&1 &
done

echo "All EL instances started."
```

### Step 5: Merge Results

```bash
#!/bin/bash
# merge_results.sh

mkdir -p outputs/energy-merged/el

# Merge all EL JSONL files
cat outputs/energy-part*/el/*.jsonl > outputs/energy-merged/el/all_linked.jsonl

echo "Merged $(wc -l < outputs/energy-merged/el/all_linked.jsonl) documents"
```

### Parallel Processing Best Practices

1. **Monitor GPU memory:** Each process needs ~4-6 GB VRAM
2. **Stagger start times:** Start processes 30 seconds apart to avoid initialization conflicts
3. **Use separate output directories:** Each process gets its own checkpoint/cache
4. **Shared cache is NOT supported:** Each parallel instance maintains its own cache
5. **Monitor logs:** `tail -f outputs/energy-part*/*.log`

---

## Performance Optimization

### Batch Size Tuning

| Setting | NER Impact | EL Impact |
|---------|------------|-----------|
| `--batch_size 500` | Slower, less memory | Fewer cache saves |
| `--batch_size 1000` (default) | Balanced | Balanced |
| `--batch_size 2000` | Faster, more memory | More cache batching |

### Linker Selection by Speed

| Priority | Linker | Speed | Accuracy |
|----------|--------|-------|----------|
| Speed | `semantic` | ⚡⚡⚡ | 🎯🎯 |
| Balance | `instruct` | ⚡⚡ | 🎯🎯🎯 |
| Accuracy | `reranker` | ⚡ | 🎯🎯🎯🎯 |

### Threshold Tuning

| Threshold | Linking Rate | Precision | Use Case |
|-----------|--------------|-----------|----------|
| 0.5 | ~95% | ~80% | High recall needed |
| 0.7 | ~85% | ~90% | Balanced |
| 0.8 (default) | ~75% | ~95% | High precision (recommended) |
| 0.9 | ~60% | ~98% | Very high precision |

### Reranker Optimization

```bash
# Fast reranker (lower accuracy)
--reranker_top_k 3 --threshold 0.6

# Balanced (default from el_config)
--reranker_top_k 7 --threshold 0.8

# High accuracy (slower)
--reranker_top_k 10 --threshold 0.85 --reranker_thinking
```

### Context Settings

| Setting | Speed | Accuracy | When to Use |
|---------|-------|----------|-------------|
| `--context_window 0` | Fastest | Lower | Entity text is unambiguous |
| `--context_window 3` | Balanced | Good | Most cases |
| `--context_window 5` (default) | Slower | Better | Ambiguous entities |
| `--use_sentence_context` | Slowest | Best | Complex disambiguation |

---

## Configuration Recipes

With `el_config`, most commands are now simplified. These recipes show how to override defaults when needed.

### Recipe 1: Default (Recommended)

Uses all settings from domain `el_config` - no overrides needed.

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step all \
    --resume
```

**Expected:** ~93% precision, ~75-85% linking rate (threshold 0.80)

### Recipe 2: High Precision

For applications where false positives are costly.

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy-highprec \
    --step el \
    --threshold 0.85 \
    --reranker_top_k 10 \
    --resume
```

**Expected:** ~95% precision, ~70% linking rate

### Recipe 3: High Recall

For applications where missing entities is costly.

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy-highrecall \
    --step el \
    --linker_type semantic \
    --threshold 0.5 \
    --resume
```

**Expected:** ~80% precision, ~95% linking rate

### Recipe 4: Maximum Speed

For large-scale processing where speed is critical.

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy-fast \
    --step el \
    --linker_type semantic \
    --threshold 0.6 \
    --context_window 0 \
    --resume
```

**Expected:** ~85% precision, ~90% linking rate, 5x faster

### Recipe 5: Development/Testing

For rapid experimentation and tuning.

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy/sample \
    --output outputs/energy_dev \
    --step all \
    --linker_type semantic \
    --threshold 0.65 \
    --batch_size 10 \
    --debug \
    --resume
```

**Tips:**
- Use small sample first (~100 docs)
- Iterate quickly with semantic linker
- Switch to reranker (default) once config is stable

### Recipe 6: Large Vocabulary (Cancer)

For domains with millions of taxonomy entries. Uses FTS5 automatically.

```bash
python src/pipeline.py \
    --domain cancer \
    --input data/cancer \
    --output outputs/cancer \
    --step all \
    --resume
```

---

## Checkpoint Management

### Checkpoint Structure

```
outputs/energy/
├── ner/
│   └── checkpoints/
│       └── processed_sections.json    # Section-level progress
├── sections/
│   └── energy_ner_sections.csv        # Section texts
└── el/
    ├── checkpoints/
    │   └── processed.json             # File-level progress
    └── cache/
        └── linking_cache.json         # Entity linking cache
```

### Resume from Checkpoint

```bash
# Interrupted run
python src/pipeline.py --domain energy --step ner --output outputs/energy

# Resume (automatically skips processed sections)
python src/pipeline.py --domain energy --step ner --output outputs/energy --resume
```

### Manual Checkpoint Inspection

```python
import json

# Check NER progress
with open("outputs/energy/ner/checkpoints/processed_sections.json") as f:
    processed = json.load(f)
print(f"Processed: {len(processed)} sections")

# Check EL progress
with open("outputs/energy/el/checkpoints/processed.json") as f:
    processed = json.load(f)
print(f"Processed: {len(processed)} files")
print("Last file:", list(processed.keys())[-1])

# Check cache size
with open("outputs/energy/el/cache/linking_cache.json") as f:
    cache = json.load(f)
    linked = sum(1 for v in cache.values() if v is not None)
    rejected = sum(1 for v in cache.values() if v is None)
    print(f"Cache: {linked} linked, {rejected} rejected")
```

### Clear Checkpoints (Start Fresh)

```bash
# Clear all checkpoints for a domain
rm -rf outputs/energy/ner/checkpoints
rm -rf outputs/energy/el/checkpoints
rm -rf outputs/energy/el/cache

# Or just remove specific checkpoint
rm outputs/energy/ner/checkpoints/processed_sections.json
```

---

## Logging Configuration

### Log Levels

```bash
# Normal mode (INFO level)
python src/pipeline.py --domain energy --step all --output outputs/energy --resume

# Debug mode (detailed logging)
python src/pipeline.py --domain energy --step all --output outputs/energy --debug --resume
```

**Debug mode shows:**
- Every entity linking decision
- LLM prompts and responses
- Cache hits/misses
- Filter decisions (blocked mentions, too short)
- Type matching results

**Info mode shows:**
- File-level progress
- Summary statistics
- Periodic cache stats

### Log File Locations

```
outputs/energy/
├── ner/
│   └── logs/
│       └── energy_ner.log
└── el/
    └── logs/
        └── energy_el.log
```

### Log Analysis

**Find rejection patterns:**
```bash
grep "REJECT" outputs/energy/el/logs/energy_el.log \
  | cut -d"'" -f2 \
  | sort | uniq -c | sort -rn | head -20
```

**Track linking rate:**
```bash
grep "linked" outputs/energy/el/logs/energy_el.log | tail -20
```

**Track linking rate over time:**
```bash
grep "entities linked" outputs/energy/el/logs/energy_el.log \
  | awk '{print $6}' \
  | cut -d'/' -f1
```

**Monitor cache performance:**
```bash
grep "Cache" outputs/energy/el/logs/energy_el.log | tail -10
```

**Check blocked entities:**
```bash
grep "blocked" outputs/energy/el/logs/energy_el.log | head -50
```

**Check entities filtered by length:**
```bash
grep "too short" outputs/energy/el/logs/energy_el.log | head -50
```

**Check type mismatches:**
```bash
grep "type mismatch" outputs/energy/el/logs/energy_el.log | head -50
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
    --batch_size 5 \
    --resume
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

### Issue: Out of Memory (OOM)

**Symptoms:** Process killed, CUDA out of memory

**Solutions:**
1. Reduce batch size: `--batch_size 500`
2. Reduce parallel instances
3. Use FTS5 instead of semantic linkers
4. Clear GPU cache between batches (automatic)

### Issue: Low Linking Rate (<60%)

**Symptoms:** Many entities not linked

**Solutions:**
1. Lower threshold: `--threshold 0.5`
2. Enable fallbacks: `--reranker_fallbacks` (default: True)
3. Check taxonomy coverage
4. Review blocked_mentions (may be too aggressive)

**Debug:**
```bash
# Run with debug logging
--debug

# Check entity types being detected
grep "entity" outputs/energy/logs/energy_ner.log | head -50

# Check why entities aren't linking
grep "below threshold" outputs/energy/logs/energy_el.log | head -20
```

### Issue: Too Many False Positives

**Symptoms:** Incorrect links

**Solutions:**
1. `use_context_for_retrieval=false` (already default)
2. Raise threshold: `--threshold 0.85`
3. Review GLiNER labels (may be too broad)
4. Add more blocked_mentions

**Debug:**
```bash
# Sample false positives for review
python tools/sample_for_review.py \
    --input outputs/energy/el \
    --filter_by high_score \
    --output review_fps.csv
```

### Issue: Slow Processing

**Symptoms:** Processing takes too long

**Solutions:**
1. Use faster linker: `--linker_type semantic`
2. Reduce context: `--context_window 0`
3. Increase batch size: `--batch_size 2000`
4. Run in parallel (see Parallel Processing section)

### Issue: Text Too Long for spaCy

**Symptoms:** `Text of length X exceeds maximum of 1000000`

**Solutions:**
- Automatic: Pipeline truncates to 1M characters
- Entities beyond truncation are skipped with warning
- This affects <0.1% of typical documents

### Issue: CSV Escape Errors

**Symptoms:** `need to escape, but no escapechar set`

**Solutions:**
- Automatic: Pipeline uses `escapechar='\\'`
- Title/abstract reader normalizes whitespace
- Already fixed in latest version

### Issue: Checkpoint Not Found

**Symptoms:** Processing restarts from beginning

**Solutions:**
1. Check `--output` path matches previous run
2. Verify checkpoint file exists:
   ```bash
   ls -la outputs/energy/ner/checkpoints/
   ```
3. Use `--resume` flag

### Issue: Cache Not Persisting

**Symptoms:** Same entities being re-linked

**Solutions:**
1. Check cache directory permissions
2. Verify cache file:
   ```bash
   ls -la outputs/energy/el/cache/
   ```
3. Cache saves after each file, not each entity

**Fix:**
```bash
# Reset cache if corrupted
rm outputs/energy/el/cache/linking_cache.json

# Verify permissions
chmod 644 outputs/energy/el/cache/linking_cache.json
```

---

## Quick Reference

### Essential Commands

```bash
# Full pipeline (uses el_config defaults)
python src/pipeline.py --domain energy --input data/energy --output outputs/energy --step all --resume

# NER only (title/abstract)
python src/pipeline.py --domain energy --input data/energy.json --output outputs/energy --input_format title_abstract --step ner --resume

# EL only (uses el_config defaults)
python src/pipeline.py --domain energy --output outputs/energy --step el --resume

# Check progress
tail -f outputs/energy/*/logs/*.log

# Check GPU
nvidia-smi
```

### Configuration Checklist

- [ ] Domain configured in `src/domain_models.py`
- [ ] `el_config` defined with appropriate settings
- [ ] Taxonomy file in `taxonomies/{domain}/`
- [ ] GLiNER labels include disambiguation categories
- [ ] Taxonomy has rich aliases and descriptions
- [ ] FTS5 indices built (if using)
- [ ] `blocked_mentions` set appropriately
- [ ] `min_mention_length` configured
- [ ] Output directory has write permissions
- [ ] Sufficient GPU memory for parallel instances

### Best Practices

1. **Use defaults:** Domain `el_config` settings are optimized
2. **Test small:** Run on 100 docs before scaling
3. **Monitor logs:** Check linking rate and cache performance
4. **Preserve cache:** Keep `linking_cache.json` between runs
5. **Use checkpoints:** Always enable `--resume`
6. **Validate regularly:** Sample for manual review
7. **Enrich taxonomy:** Add Wikidata aliases for better coverage

### Shell Scripts

Pre-configured shell scripts are available for common scenarios:

```bash
# NIF format (full pipeline)
./run_energy_ner_el.sh
./run_neuro_ner_el.sh
./run_ccam_ner_el.sh
./run_maritime_ner_el.sh
./run_cancer_ner_el.sh

# Title/abstract (parallel processing)
./run_energy_ner_parallel_titles-abstracts.sh
./run_energy_el_parallel_titles-abstracts.sh

# Legal text
./run_energy_ner_legal.sh
./run_energy_el_legal.sh
```

---

## Summary

This configuration guide covers:

✅ **Domain setup** with multi-label NER configuration
✅ **el_config** for centralized entity linking configuration
✅ **Entity filtering** with blocked mentions and length thresholds
✅ **Taxonomy preparation** with best practices and Wikidata enrichment
✅ **Environment setup** with hardware recommendations
✅ **Parallel processing** for large-scale deployments
✅ **Performance optimization** through threshold and batch tuning
✅ **Configuration recipes** for different precision/recall trade-offs
✅ **Checkpoint management** for reliable resumption
✅ **Logging and debugging** with analysis commands
✅ **Testing and validation** with quality checks
✅ **Troubleshooting** for common issues

For more information, see:
- [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - System architecture
- [ENTITY_LINKING_README.md](ENTITY_LINKING_README.md) - Linking strategies
- [RERANKER_GUIDE.md](RERANKER_GUIDE.md) - RerankerLinker deep dive
