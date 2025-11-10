# SciLake NER & Entity Linking Pipeline

Python pipeline for extracting domain-specific entities from scientific literature (NIF/RDF format) and linking them to controlled vocabularies.

**Supported domains**: Energy, Neuro, CCAM (transport), Maritime, Cancer

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/sirisacademic/scilake-enrichments.git
cd scilake-enrichments

# Create environment (Conda recommended)
conda env create -f environment.yml
conda activate scilake-enrichments

# OR use pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### Basic Usage

```bash
# Run full pipeline (NER + Entity Linking)
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step all \
    --resume
```

---

## 📋 Pipeline Overview

### 🔄 Workflow

1. **Extract text** from NIF `.ttl` files in `data/{domain}/*.ttl`
2. **Expand acronyms** using SciSpacy (Schwartz-Hearst algorithm)
3. **Detect entities** with domain-specific NER models:
   - GLiNER (multi-label semantic matching)
   - RoBERTa (fine-tuned for domain)
   - Gazetteer (exact matching against taxonomy)
4. **Link entities** to controlled vocabularies:
   - Domain-specific taxonomies (IRENA, openMINDS, etc.)
   - Wikidata for additional context
5. **Export enriched outputs** to `outputs/{domain}/`

### Two-Stage Process

#### Stage 1: Named Entity Recognition (NER)

**Components:**
- **Acronym Expansion**: Uses SciSpacy to detect and expand abbreviations (e.g., "PV" → "photovoltaic")
- **Gazetteer Matching**: FlashText-based exact string matching against taxonomy terms
- **Neural NER**: 
  - **GLiNER**: Multi-label semantic matching (gives model options for ambiguous entities)
  - **RoBERTa**: Domain-specific fine-tuned token classification
- **Entity Merging**: Deduplicates and resolves overlaps

**Output**: Detected entities with character offsets in `.jsonl` format

#### Stage 2: Entity Linking (EL)

**Components:**
- **Four linking strategies** (choose based on speed/accuracy needs):
  1. **Gazetteer Linker**: Exact matches only (runs automatically during NER)
  2. **Semantic Linker**: Fast embedding-based similarity
  3. **Instruct Linker**: Instruction-tuned embeddings
  4. **Reranker Linker** ⭐: Two-stage (embedding + LLM reranking) - **Recommended**

**Output**: Entities enriched with taxonomy IDs and Wikidata links

---

## 🔧 Configuration

### NER Step

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step ner \
    --batch_size 100 \
    --resume
```

**Parameters:**

| Flag | Description | Default |
|------|-------------|---------|
| `--domain` | Domain name (energy, neuro, ccam, maritime, cancer) | Required |
| `--input` | Directory with `.ttl` NIF files (recursive search) | Required |
| `--output` | Output directory for results | Required |
| `--step` | Pipeline step: `ner` \| `el` \| `all` | `ner` |
| `--batch_size` | Files per batch (for checkpointing) | 1000 |
| `--resume` | Resume from checkpoint if interrupted | Flag |

**🧬 Run NER**

```bash
python -m src.pipeline --domain energy --input data/energy --output outputs/energy --step ner --batch_size 8
```

**🌍 Run Geotagging**

Example:
```bash
python -m src.pipeline \
  --domain energy \
  --input data/energy \
  --output outputs/energy \
  --step geotagging \
  --batch_size 8
```
You can then link geotagged outputs:
```
python -m src.geo_linker \
  --input_dir outputs/energy/geotagging-ner \
  --output_dir outputs/energy/geotagging-linked

```

**🏛️ Run Affiliation Enrichment (AffilGood)**
```bash
python -m src.pipeline --domain energy --input data/energy_v2 --output outputs/energy --step affiliations --batch_size 8
```

The pipeline automatically:
- ✅ Resumes from checkpoints (`outputs/<domain>/checkpoints/processed.json`)
- ✅ Logs progress under `outputs/<domain>/logs/`
- ✅ Skips already-processed files
- ✅ Saves partial results safely on interruption

---

### Entity Linking Step

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --linker_type reranker \
    --threshold 0.7 \
    --context_window 3 \
    --resume
```

**Core Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--linker_type` | Linking strategy: `auto` \| `semantic` \| `instruct` \| `reranker` | `auto` |
| `--threshold` | Similarity threshold (0.0-1.0) | 0.7 |
| `--context_window` | Context size in tokens (0 = no context) | 3 |
| `--max_contexts` | Max contexts per entity | 3 |
| `--use_sentence_context` | Use full sentences vs token windows | Flag |
| `--taxonomy` | Path to taxonomy TSV file | `taxonomies/energy/IRENA.tsv` |
| `--taxonomy_source` | Taxonomy name for output | `IRENA` |

**Reranker-Specific Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--reranker_llm` | LLM model for reranking | `Qwen/Qwen3-1.7B` |
| `--reranker_top_k` | Number of candidates | 5 |
| `--reranker_fallbacks` | Add top-level categories | True |
| `--use_context_for_retrieval` | Use context in Stage 1 (embedding) | False |
| `--reranker_thinking` | Enable chain-of-thought (slower) | False |

---

## 🔗 Entity Linking Strategies

The pipeline offers **four linking strategies** with different speed/accuracy trade-offs:

| Linker | Speed | Accuracy | GPU Required | Best For |
|--------|-------|----------|--------------|----------|
| **Gazetteer** | ⚡⚡⚡ | 🎯🎯🎯 | No | Exact matches only |
| **Semantic** | ⚡⚡ | 🎯🎯 | Optional | Large-scale, CPU-only |
| **Instruct** | ⚡ | 🎯🎯🎯 | Optional | Balanced speed/accuracy |
| **Reranker** ⭐ | 🐢 | 🎯🎯🎯🎯 | Yes (LLM) | Highest quality |

### Recommended: Reranker Linker

The **RerankerLinker** uses a two-stage approach for optimal accuracy:

```bash
python src/pipeline.py \
    --domain energy \
    --step el \
    --linker_type reranker \
    --threshold 0.7 \
    --use_context_for_retrieval false \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 5
```

**Key features:**
- 🎯 Best accuracy (~93% precision)
- 🚫 Can REJECT non-domain entities
- 🧠 Context-aware LLM reasoning
- ⚡ Fast candidate retrieval + careful reranking

**Critical setting**: `--use_context_for_retrieval false` prevents context contamination in Stage 1 while Stage 2 still uses context for validation.

**See detailed comparison and configuration:** [ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)  
**Deep dive into RerankerLinker:** [RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)

---

## 💾 Large-Scale Processing (20k+ Documents)

The pipeline is optimized for large-scale processing:

### Key Features

1. **Checkpointing**: Resume from interruptions
   ```bash
   # Interrupted run
   python src/pipeline.py --domain energy --step all --output outputs/energy
   
   # Resume (automatically skips processed files)
   python src/pipeline.py --domain energy --step all --output outputs/energy --resume
   ```

2. **Cache Persistence**: Linking decisions saved to disk
   - Cache warms up over time (~80% hit rate after 1000 docs)
   - Dramatically improves speed for recurring entities
   - Preserved across runs

3. **Progress Tracking**: Detailed logging with tqdm
   ```
   Processing files: 100%|████████| 1000/1000 [00:45:23<00:00, 22.1 files/s]
   ```

4. **Batch Processing**: Configurable batch size
   ```bash
   --batch_size 100  # Process 100 files per batch
   ```

### Monitoring Progress

**Cache statistics** (every 500 files):
```
📊 Cache stats: 5234 total, 4456 linked (85.1%), 778 rejected (14.9%)
```

**Cache checkpoints** (every 100 files):
```
💾 Cache checkpoint saved: 5234 entries
```

**Final summary**:
```
🎉 Entity Linking complete!
📊 Total: 15234 entities processed
✅ Linked: 12987 (85.2%)
❌ Rejected: 2247 (14.8%)
⚡ Avg time: 68ms per entity
💾 Cache hit rate: 82.3%
```

---

## 🧩 Domain Configuration

### Supported Domains & Knowledge Bases

| Domain | Knowledge Base | Notes |
|--------|----------------|-------|
| **Energy** | IRENA | Renewable energy taxonomy (~9000 concepts) |
| **Neuro** | openMINDS, UBERON | CNS-limited neuroanatomy |
| **CCAM** | Project-specific | Transport/mobility concepts |
| **Maritime** | Project taxonomy | Maritime domain terms |
| **Cancer** | NCBI, DO, MeSH, DrugBank | Biomedical baseline |

**Fallback**: Wikification for unknown entities

### Configuring Domains

Edit `configs/domain_models.py`:

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
                    "transportation",  # For disambiguation!
                    "measurement unit"
                ]
            },
            "roberta": {
                "enabled": True,
                "model_name": "SIRIS-Lab/SciLake-Energy-roberta-base"
            }
        },
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv"
        },
        "entity_linking": {
            "default_linker": "reranker",
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "threshold": 0.7
        }
    }
}
```

**Key principle**: Multi-label NER configuration is critical for GLiNER. Providing alternative categories (like "transportation" for energy domain) dramatically improves accuracy by helping the model reject ambiguous entities.

---

## 📦 NIF Format

### Input Format

NIF (NLP Interchange Format) is an RDF-based format for representing text and annotations.

**Example input** (`data/energy/paper1.ttl`):

```turtle
@prefix nif: <http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#> .
@prefix dct: <http://purl.org/dc/terms/> .

<http://scilake.eu/resource#context_1>
    a nif:Context ;
    nif:isString "Wind turbines convert kinetic energy into electricity." .

<http://scilake.eu/resource#section_1>
    a nif:Section ;
    nif:referenceContext <http://scilake.eu/resource#context_1> ;
    dct:title "Introduction" ;
    nif:anchorOf "Wind turbines convert kinetic energy into electricity." .
```

### Output Format

Enriched NIF files include entity annotations with taxonomy links:

**Example output** (`outputs/energy/paper1.ttl`):

```turtle
<http://scilake.eu/resource#offset_0_13>
    a nif:EntityOccurrence ;
    nif:referenceContext <http://scilake.eu/resource#context_1> ;
    nif:beginIndex "0"^^xsd:int ;
    nif:endIndex "13"^^xsd:int ;
    nif:anchorOf "Wind turbines" ;
    itsrdf:taIdentRef <http://irena.org/kb/230000> ;
    itsrdf:taIdentRef <http://www.wikidata.org/entity/Q43302> .
```

**JSONL intermediate format** (`outputs/energy/ner/paper1.jsonl`):

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
        "method": "reranker"
      }
    }
  ]
}
```

---

## 📂 Repository Structure

```
scilake-enrichments/
│
├── configs/
│   └── domain_models.py         # Domain-specific model configurations
│
├── data/
│   ├── energy/
│   ├── neuro/
│   ├── ccam/
│   ├── maritime/
│   └── cancer/
│       └── *.ttl                # Input NIF files
│
├── docs/
│   ├── ARCHITECTURE_OVERVIEW.md # System architecture
│   ├── ENTITY_LINKING_README.md # Linking strategies guide
│   ├── RERANKER_GUIDE.md        # RerankerLinker deep dive
│   └── CONFIGURATION_GUIDE.md   # Configuration recipes
│
├── outputs/
│   └── {domain}/
│       ├── ner/                 # NER results (.jsonl)
│       │   ├── checkpoints/
│       │   └── logs/
│       ├── el/                  # Entity linking results
│       │   ├── *.jsonl
│       │   ├── cache/           # Persistent linking cache
│       │   ├── checkpoints/
│       │   └── logs/
│       └── enriched/            # Final enriched .ttl files
│
├── src/
│   ├── pipeline.py              # Main orchestrator
│   ├── ner_runner.py            # NER inference logic
│   ├── nif_reader.py            # NIF parsing & acronym expansion
│   ├── gazetteer_linker.py      # Exact string matching
│   ├── semantic_linker.py       # Semantic similarity linking
│   ├── instruct_linker.py       # Instruction-based linking
│   ├── reranker_linker.py       # Two-stage reranking
│   └── utils/
│       ├── io_utils.py
│       └── logger.py
│
├── taxonomies/
│   └── {domain}/
│       └── *.tsv                # Domain taxonomies (IRENA, etc.)
│
├── environment.yml              # Conda environment
├── requirements.txt             # Pip dependencies
├── LICENSE
└── README.md                    # This file
```

---

## 🐛 Troubleshooting

### Quick Diagnostics

| Problem | Quick Fix | More Info |
|---------|-----------|-----------|
| Low linking rate (<80%) | `--threshold 0.6` or `--linker_type reranker` | [ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md#troubleshooting) |
| Too many false positives | `--use_context_for_retrieval false` (Reranker) | [RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md#troubleshooting) |
| Slow processing | `--linker_type semantic` or wait for cache warmup | [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md) |
| Out of memory | Run `--step ner` and `--step el` separately | [README.md](#out-of-memory-gpu) |
| Pipeline interrupted | Add `--resume` flag | [README.md](#pipeline-interrupted) |

### Common Issues

<details>
<summary><b>Low Linking Rate (<80%)</b></summary>

```bash
# Try lowering threshold
--threshold 0.6

# Or use more powerful linker
--linker_type reranker

# Enable debug logging to see what's happening
--debug
```

Check logs: `grep "below threshold" outputs/energy/logs/energy_el.log`
</details>

<details>
<summary><b>Too Many False Positives</b></summary>

For RerankerLinker:
```bash
# Disable context in Stage 1 (most important!)
--use_context_for_retrieval false

# Raise threshold
--threshold 0.8

# Reduce candidates
--reranker_top_k 3
```

Check rejections: `grep "REJECT" outputs/energy/logs/energy_el.log | head -20`
</details>

<details>
<summary><b>Slow Processing</b></summary>

```bash
# Use faster linker
--linker_type semantic  # or instruct

# Reduce context
--context_window 2

# Note: First 500 docs are slower (cold cache)
# Performance improves dramatically after cache warms up
```

Monitor cache: `grep "Cache stats" outputs/energy/logs/energy_el.log`
</details>

<details>
<summary><b>Out of Memory (GPU)</b></summary>

```bash
# Run stages separately to free memory
python src/pipeline.py --domain energy --step ner --output outputs/energy
python src/pipeline.py --domain energy --step el --output outputs/energy

# Use smallest LLM
--reranker_llm Qwen/Qwen3-1.7B

# Reduce batch size
--batch_size 50
```

**Memory requirements:** NER: ~3GB | EL (Reranker): ~5-8GB
</details>

<details>
<summary><b>Pipeline Interrupted</b></summary>

```bash
# Simply add --resume flag
python src/pipeline.py \
    --domain energy \
    --step all \
    --output outputs/energy \
    --resume
```

The pipeline automatically:
- ✅ Loads checkpoint from `outputs/energy/checkpoints/processed.json`
- ✅ Skips already-processed files  
- ✅ Preserves existing cache
</details>

**For detailed troubleshooting guides, see:**
- [ENTITY_LINKING_README.md - Troubleshooting Section](docs/ENTITY_LINKING_README.md#troubleshooting)
- [RERANKER_GUIDE.md - Troubleshooting Section](docs/RERANKER_GUIDE.md#troubleshooting)
- [CONFIGURATION_GUIDE.md - Common Issues](docs/CONFIGURATION_GUIDE.md#troubleshooting-common-issues)

---

## 📊 Evaluation & Quality Control

### Expected Performance

| Metric | Target | Typical (Energy + Reranker) |
|--------|--------|------------------------------|
| NER Precision | >90% | ~92% |
| NER Recall | >85% | ~87% |
| Linking Precision | >90% | ~93% |
| Linking Rate | >80% | ~85% |
| Cache Hit Rate | >70% (after 100 docs) | ~80% |
| Throughput | >100 entities/sec | ~150 entities/sec (warm) |

### Monitoring Quality

**Check rejection rates** in logs:
```bash
grep "rejected" outputs/energy/logs/energy_el.log | tail -50
```

**Typical rejection rates:**
- Energy: 10-20% (chemicals, emissions should be rejected)
- Too high (>30%): Threshold might be too strict
- Too low (<5%): May be linking non-domain entities

**Cache statistics:**
```bash
grep "Cache stats" outputs/energy/logs/energy_el.log
```

### Manual Review

Sample outputs for validation:

```python
# tools/sample_for_review.py
python tools/sample_for_review.py \
    --input outputs/energy/el/ \
    --output review_sample.csv \
    --n 100
```

Review:
1. Entity detection quality (correct spans + labels)
2. Linking accuracy (correct taxonomy IDs)
3. Rejection decisions (appropriate filtering)

---

## 📚 Documentation

This README provides a quick start and overview. For detailed information:

### Core Documentation

| Document | Contents | When to Read |
|----------|----------|--------------|
| **[ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)** | System architecture, data flow, component details, performance characteristics | Understanding how the system works internally |
| **[ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)** | Complete guide to all 4 linking approaches, when to use each, configuration examples | Choosing and configuring a linker |
| **[RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)** | Deep dive into RerankerLinker: two-stage architecture, prompt engineering, optimization | Using the recommended approach for production |
| **[CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)** | Configuration recipes (high precision, high recall, speed, balanced), taxonomy preparation, testing | Setting up for your specific use case |

### Quick Links by Topic

**Getting Started:**
- Installation & Quick Start → This README
- Understanding the pipeline → [ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)

**Choosing a Linker:**
- Overview of options → This README ([Entity Linking Strategies](#-entity-linking-strategies))
- Detailed comparison → [ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)
- RerankerLinker specifics → [RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)

**Configuration:**
- Basic parameters → This README ([Configuration](#-configuration))
- Recipe examples → [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)
- Domain setup → [CONFIGURATION_GUIDE.md - Domain Configuration](docs/CONFIGURATION_GUIDE.md#domain-configuration)

**Optimization:**
- Performance tuning → [ENTITY_LINKING_README.md - Performance Tuning](docs/ENTITY_LINKING_README.md#performance-tuning)
- RerankerLinker optimization → [RERANKER_GUIDE.md - Performance Optimization](docs/RERANKER_GUIDE.md#performance-optimization)
- Large-scale processing → [CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)

**Troubleshooting:**
- Common issues → This README ([Troubleshooting](#-troubleshooting))
- Linking issues → [ENTITY_LINKING_README.md - Troubleshooting](docs/ENTITY_LINKING_README.md#troubleshooting)
- RerankerLinker issues → [RERANKER_GUIDE.md - Troubleshooting](docs/RERANKER_GUIDE.md#troubleshooting)

---

## 🤝 Contributing

When adding new domains:

1. Add model configuration to `configs/domain_models.py`
2. Create taxonomy file in `taxonomies/{domain}/`
3. Test on small sample (100-500 docs) before full processing
4. Document domain-specific considerations
5. Submit PR with evaluation results

---

## 📄 License

MIT

---

## 💡 Support

For issues or questions:

1. **Check logs**: `outputs/{domain}/*/logs/`
2. **Enable debug mode**: `--debug`
3. **Review cache**: Inspect linking decisions
4. **Test small sample**: 100-500 documents first
5. **Consult documentation**: See `docs/` directory
6. **Open an issue**: [GitHub Issues](https://github.com/sirisacademic/scilake-enrichments/issues)

---

## 🙏 Acknowledgments

This work is part of the **SciLake Project**, which aims to create a coherent ecosystem for Open Science scholarly communication.

**Key technologies:**
- [GLiNER](https://github.com/urchade/GLiNER) - Multi-label NER
- [Sentence Transformers](https://www.sbert.net/) - Semantic embeddings
- [Qwen](https://github.com/QwenLM/Qwen) - LLM for reranking
- [SciSpacy](https://allenai.github.io/scispacy/) - Scientific text processing
- [NIF](http://persistence.uni-leipzig.org/nlp2rdf/) - RDF format for NLP


