# SciLake NER & Entity Linking Pipeline

Python pipeline for extracting domain-specific entities from scientific literature and linking them to controlled vocabularies.

**Supported domains**: Energy, Neuro, CCAM (transport), Maritime, Cancer

**Supported input formats**: NIF/RDF files, Title/Abstract JSON, Legal Text JSON

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

## 📥 Input Formats

The pipeline supports three input formats:

| Format | Flag | Description | Use Case |
|--------|------|-------------|----------|
| `nif` (default) | `--input_format nif` | NIF/RDF TTL files | Full-text scientific papers |
| `title_abstract` | `--input_format title_abstract` | JSON with title/abstract | OpenAIRE metadata |
| `legal_text` | `--input_format legal_text` | JSON with legal documents | Regulatory texts (e.g., Fedlex) |

### NIF Format (Default)

Standard NIF/RDF format with `.ttl` files:

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step all
```

### Title/Abstract JSON Format

For processing publication metadata (titles and abstracts):

**Input format** (JSONL - one JSON object per line):
```json
{
    "oaireid": "50|doi_dedup___::abc123",
    "titles": ["Paper Title Here"],
    "abstracts": ["Abstract text here..."],
    "pids": [{"scheme": "doi", "value": "10.1234/example"}]
}
```

**Usage:**
```bash
python src/pipeline.py \
    --domain energy \
    --step ner \
    --input_format title_abstract \
    --input data/energy_titleabstract.json \
    --output outputs/energy-titleabstract \
    --resume
```

**Features:**
- Combines title and abstract into single section (faster processing)
- Saves results incrementally after each batch
- Supports resume from checkpoint

### Legal Text JSON Format

For processing legal/regulatory documents:

**Input format** (JSONL):
```json
{
    "rsNr": "0.101",
    "en_lawTitle": "Convention for the Protection of Human Rights...",
    "en_lawText": "Full legal text content..."
}
```

**Usage:**
```bash
python src/pipeline.py \
    --domain energy \
    --step ner \
    --input_format legal_text \
    --input data/fedlex-dataset.jsonl \
    --output outputs/energy-legal \
    --resume
```

**Features:**
- Handles very long documents (automatic chunking in NER)
- Truncates sections >1M chars for entity linking (spaCy limit)
- Normalizes whitespace in legal texts

---

## 📋 Pipeline Overview

### 🔄 Workflow

1. **Extract text** from input files (NIF, JSON title/abstract, or legal text)
2. **Expand acronyms** using SciSpacy (Schwartz-Hearst algorithm)
3. **Detect entities** with domain-specific NER models:
   - GazetteerLinker (exact matching - extracts AND links in one step)
   - GLiNER (multi-label semantic matching)
   - RoBERTa (fine-tuned for domain)
4. **Filter entities** using domain-level blocked mentions and minimum length
5. **Link entities** to controlled vocabularies:
   - Domain-specific taxonomies (IRENA, openMINDS, etc.)
   - Wikidata for additional context
6. **Export enriched outputs** to `outputs/{domain}/`

### Two-Stage Process

#### Stage 1: Named Entity Recognition (NER)

**Components:**
- **Acronym Expansion**: Uses SciSpacy to detect and expand abbreviations (e.g., "PV" → "photovoltaic")
- **GazetteerLinker**: FlashText-based exact matching - **extracts AND links** entities found in taxonomy (non-cancer domains only)
- **Neural NER**: 
  - **GLiNER**: Multi-label semantic matching (gives model options for ambiguous entities)
  - **RoBERTa**: Domain-specific fine-tuned token classification
  - **AIOner**: Biomedical NER (cancer domain)
- **Entity Merging**: Deduplicates and resolves overlaps (Gazetteer has priority)

**Output**: Detected entities with character offsets in `.jsonl` format. Gazetteer-found entities are already linked.

#### Stage 2: Entity Linking (EL)

**Purpose**: Link entities NOT already linked by GazetteerLinker.

**Components:**
- **FTS5Linker** ⭐: SQLite exact matches (disk-based, used for cancer domain)
- **SemanticLinker**: Fast embedding-based similarity
- **InstructLinker**: Instruction-tuned embeddings
- **RerankerLinker** ⭐: Two-stage (embedding + LLM reranking) - **Best accuracy**

**Output**: Entities enriched with taxonomy IDs and Wikidata links

### Domain-Specific Architectures

The pipeline uses different architectures depending on domain characteristics:

| Domain | NER Step | EL Step | Why |
|--------|----------|---------|-----|
| **Energy, Neuro, CCAM, Maritime** | GazetteerLinker + Neural NER | Semantic/Reranker | Small taxonomies, low ambiguity |
| **Cancer** | Neural NER only (no Gazetteer) | FTS5Linker | Large taxonomies, high ambiguity |

**Why the difference?** Cancer domain has large, ambiguous vocabularies (millions of gene symbols like "MET", "ALL", "CAT"). Scanning text with a gazetteer would produce too many false positives. Instead, a specialized NER model extracts entities contextually, then FTS5 links them.

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
| `--input` | Directory with `.ttl` NIF files or JSON file | Required |
| `--output` | Output directory for results | Required |
| `--input_format` | Input format: `nif` \| `title_abstract` \| `legal_text` | `nif` |
| `--step` | Pipeline step: `ner` \| `el` \| `geotagging` \| `affiliations` \| `all` | `ner` |
| `--batch_size` | Sections per batch (for checkpointing) | 1000 |
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

### Entity Filtering Configuration

Configure entity filtering in `configs/domain_models.py`:

```python
"energy": {
    # Minimum mention length (characters)
    "min_mention_length": 2,  # Global setting
    # OR per entity type:
    # "min_mention_length": {"gene": 2, "disease": 3, "_default": 2},
    
    # Blocked mentions (terms to skip)
    "blocked_mentions": {"energy", "power", "system", "data"},
    # OR per entity type:
    # "blocked_mentions": {
    #     "species": {"patient", "patients"},
    #     "disease": {"pain", "fever"},
    # },
}
```

---

## 🔗 Entity Linking Strategies

### Important: Extraction vs Linking

The pipeline has two types of "linkers" that serve different purposes:

| Component | Stage | Purpose | Scans Text? |
|-----------|-------|---------|-------------|
| **GazetteerLinker** | NER | Extraction + Linking | Yes |
| **FTS5/Semantic/Reranker** | EL | Linking only | No |

**GazetteerLinker** finds entities in text AND links them. **EL linkers** only link entities already found by NER.

### Linking Strategies Comparison

| Linker | Stage | Speed | Accuracy | GPU | Best For |
|--------|-------|-------|----------|-----|----------|
| **GazetteerLinker** | NER | ⚡⚡⚡ | 🎯🎯🎯 | No | Taxonomy-driven discovery |
| **FTS5Linker** ⭐ | EL | ⚡⚡⚡ | 🎯🎯🎯 | No | Large vocabularies (cancer) |
| **SemanticLinker** | EL | ⚡⚡ | 🎯🎯 | Optional | Large-scale, CPU-only |
| **InstructLinker** | EL | ⚡ | 🎯🎯🎯 | Optional | Balanced speed/accuracy |
| **RerankerLinker** ⭐ | EL | 🐢 | 🎯🎯🎯🎯 | Yes (LLM) | Highest quality |

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

### FTS5Linker (Cancer Domain)

For domains with large, ambiguous vocabularies:

```python
# In domain_models.py
"cancer": {
    "gazetteer": {"enabled": False},  # No extraction
    "linking_strategy": "fts5",
    "fts5_linkers": {
        "gene": {
            "index_path": "indices/cancer/ncbi_gene.db",
            "taxonomy_source": "NCBI_Gene",
        },
        ...
    }
}
```

**Features:**
- ✅ Disk-based (no memory issues)
- ✅ Text normalization (Greek letters → Latin: "IFN-γ" → "IFNG")
- ✅ Frequency-based disambiguation
- ✅ Scales to millions of entries

**See detailed comparison and configuration:** [ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)  
**Deep dive into RerankerLinker:** [RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)

---

## 🚀 Large-Scale & Parallel Processing (20k+ Documents)

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

3. **Incremental Saving**: Results saved after each batch
   - Results available immediately (don't have to wait for completion)
   - No data loss on crash
   - Safe to stop and resume at any time

### Parallel Processing

For large datasets (millions of records), split input files and run in parallel:

**Step 1: Split Input File**
```bash
# Split into 6 parts
split -n l/6 -d --additional-suffix=.json \
    data/energy_titleabstract.json \
    data/energy_titleabstract_part
```

**Step 2: Run Parallel NER**
```bash
for i in 00 01 02 03 04 05; do
    nohup python src/pipeline.py \
        --domain energy \
        --step ner \
        --input_format title_abstract \
        --input data/energy_titleabstract_part${i}.json \
        --output outputs/energy-part${i} \
        --resume \
        > outputs/energy-part${i}_ner.log 2>&1 &
done
```

**Step 3: Run Parallel EL**
```bash
for i in 00 01 02 03 04 05; do
    nohup python src/pipeline.py \
        --domain energy \
        --step el \
        --input_format title_abstract \
        --output outputs/energy-part${i} \
        --linker_type reranker \
        --threshold 0.70 \
        --reranker_llm Qwen/Qwen3-1.7B \
        --reranker_top_k 7 \
        --reranker_fallbacks \
        --resume \
        > outputs/energy-part${i}_el.log 2>&1 &
done
```

**Step 4: Merge Results**
```bash
# Merge NER outputs
cat outputs/energy-part*/ner/*.jsonl > outputs/energy-merged/ner/energy_ner.jsonl

# Merge sections (header from first, data from all)
head -1 outputs/energy-part00/sections/*.csv > outputs/energy-merged/sections/sections.csv
tail -n +2 -q outputs/energy-part*/sections/*.csv >> outputs/energy-merged/sections/sections.csv
```

**Monitor progress:**
```bash
tail -3 outputs/energy-part*_ner.log
nvidia-smi
```

---

## 📁 Output Format

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
├── indices/                     # FTS5 SQLite indices (cancer domain)
│   └── cancer/
│       ├── ncbi_gene.db
│       ├── doid_disease.db
│       └── ...
│
├── outputs/
│   └── {domain}/
│       ├── ner/                 # NER results (.jsonl)
│       │   ├── checkpoints/
│       │   └── logs/
│       ├── sections/            # Section texts for EL context
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
│   ├── title_abstract_reader.py # Title/abstract JSON reader
│   ├── legal_text_reader.py     # Legal text JSON reader
│   ├── gazetteer_linker.py      # Extraction + Linking (NER step)
│   ├── fts5_linker.py           # Linking only (EL step, cancer)
│   ├── build_fts5_indices.py    # Build FTS5 indices
│   ├── semantic_linker.py       # Semantic similarity linking
│   ├── instruct_linker.py       # Instruction-based linking
│   ├── reranker_linker.py       # Two-stage reranking
│   ├── geo_linker.py            # Geotagging linker
│   ├── geotagging_runner.py     # Geotagging runner
│   ├── affilgood_runner.py      # Affiliation enrichment
│   └── utils/
│       ├── io_utils.py
│       └── logger.py
│
├── taxonomies/
│   └── {domain}/
│       └── *.tsv                # Domain taxonomies (IRENA, etc.)
```

---

## 🔍 Troubleshooting

### Quick Reference

| Issue | Solution | Details |
|-------|----------|---------|
| Low linking rate | Lower threshold, enable fallbacks | [Low Linking Rate](#low-linking-rate-80) |
| Too many false positives | Disable context in retrieval, raise threshold | [Too Many False Positives](#too-many-false-positives) |
| Slow processing | Use faster linker, reduce context | [Slow Processing](#slow-processing) |
| Out of memory | Smaller batches, run stages separately | [Out of Memory](#out-of-memory-gpu) |
| Pipeline interrupted | Add `--resume` flag | [Pipeline Interrupted](#pipeline-interrupted) |
| Segmentation faults | Process smaller batches (Gazetteer issue) | [Segmentation Faults](#segmentation-faults-gazetteer-at-scale) |
| Text too long | Automatic truncation (see warning) | [Text Too Long](#text-too-long-for-spacy) |

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

<details>
<summary><b>Segmentation Faults (Gazetteer at Scale)</b></summary>

This occurs with FlashText + pandas at scale (~300+ files).

**Solutions:**
1. Process in smaller batches: `--batch_size 50`
2. Use Python parser for pandas (in code): `engine='python'`
3. For large vocabularies: consider cancer-style architecture (NER → FTS5)

**Note:** This is a known limitation of the FlashText-based GazetteerLinker. See [LINKER_ARCHITECTURE_TRACKER.md](docs/LINKER_ARCHITECTURE_TRACKER.md) for details.
</details>

<details>
<summary><b>Text Too Long for spaCy</b></summary>

**Symptoms:** `Text of length X exceeds maximum of 1000000`

**Automatic handling:**
- The pipeline automatically truncates sections >1M characters
- Entities beyond truncation point are skipped with a warning
- This affects <0.1% of typical documents

**Manual check:**
```bash
grep "Truncating section" outputs/energy/el/logs/*.log
```
</details>

<details>
<summary><b>CSV Escape Errors</b></summary>

**Symptoms:** `need to escape, but no escapechar set`

**Already fixed:** The pipeline uses `escapechar='\\'` for CSV output and normalizes whitespace in title/abstract and legal text readers.
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
| **[ENTITY_LINKING_README.md](docs/ENTITY_LINKING_README.md)** | Complete guide to linking approaches, when to use each, configuration examples | Choosing and configuring a linker |
| **[RERANKER_GUIDE.md](docs/RERANKER_GUIDE.md)** | Deep dive into RerankerLinker: two-stage architecture, prompt engineering, optimization | Using the recommended approach for production |
| **[CONFIGURATION_GUIDE.md](docs/CONFIGURATION_GUIDE.md)** | Configuration recipes (high precision, high recall, speed, balanced), taxonomy preparation, testing | Setting up for your specific use case |
| **[LINKER_ARCHITECTURE_TRACKER.md](docs/LINKER_ARCHITECTURE_TRACKER.md)** | Internal notes on Gazetteer vs FTS5 design decisions, known limitations, future plans | Understanding architectural choices |

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
3. If using FTS5, build index: `python src/build_fts5_indices.py --taxonomy ... --output ... --source ...`
4. Test on small sample (100-500 docs) before full processing
5. Document domain-specific considerations
6. Submit PR with evaluation results

---

## 📄 License

Apache-2.0

---

## 👋 Support

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
- [SQLite FTS5](https://www.sqlite.org/fts5.html) - Full-text search
- [NIF](http://persistence.uni-leipzig.org/nlp2rdf/) - RDF format for NLP
