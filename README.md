# SciLake NER & Entity Linking Enrichment

Python repository to extract entities and enrich SciLake pilot data (NIF format) with Named Entity Recognition (NER) and Entity Linking (NEL).  
This repository supports all SciLake pilots (Neuro, Energy, CCAM, Maritime, Cancer) using domain-specific models and vocabularies.

---

## 🚀 Overview

### 🔄 Workflow

1. Extract text sections from NIF `.ttl` files (`data/{domain}/*.ttl`)
2. Expand acronyms and normalize text
3. Apply **domain-specific NER models** (GLiNER, RoBERTa, AIObioEnts)
4. Enrich the original NIF files with detected entities
5. Apply **Entity Linking** (NEL) using domain KBs
6. Export enriched `.ttl` outputs to `outputs/{domain}/`

### 📁 Repository Structure
```txt
scilake-ner-enrichment/
│
├── data/
│   ├── neuro/
│   ├── energy/
│   ├── maritime/
│   ├── ccam/
│   └── cancer/
│       └── *.ttl                # Input NIF files (raw)
│
├── outputs/
│   ├── neuro/
│   ├── energy/
│   ├── maritime/
│   ├── ccam/
│   └── cancer/
│       └── *.ttl                # Enriched NIF files with NER/NEL annotations
│
├── configs/
│   └── domain_models.py         # DOMAIN_MODELS dict and label configs
│
├── notebooks/
│   └── exploration.ipynb        # Optional: dev/test notebooks
│
├── src/
│   ├── pipeline.py              # Main orchestrator for NER/NEL pipeline
│   ├── ner_runner.py            # Model inference and batching logic
│   ├── nif_reader.py            # Load and parse NIF files
│   ├── utils/                   # Logging, IO, helper functions
│
├── requirements.txt
│
├── README.md
```

---

## 🧩 Domain Models

See [`configs/domain_models.py`](configs/domain_models.py) for model definitions.

Each domain defines its models (GLiNER / RoBERTa) and thresholds, for example:

```python
DOMAIN_MODELS = {
    "neuro": {
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Neuroscience-GLiNER-large",
                "type": "gliner",
                "threshold": 0.99,
            },
        ],
        "labels": {
            "gliner": [
                "UBERONParcellation",
                "species",
                "preparationType",
                "technique",
                "biologicalSex",
            ],
        },
        "kb": {
            "UBERONParcellation": "UBERON (restricted to CNS)",
            "others": "openMINDS controlled terms",
        },
    },
    ...
```
---
## ⚙️ Installation

```bash
git clone https://github.com/sirisacademic/scilake-enrichments.git
cd scilake-enrichments

# Create environment
conda env create -f environment.yml
conda activate scilake-enrichments

# or, use pip
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download required SpaCy model
python -m spacy download en_core_web_sm

```

## 🧠 Running the Pipeline

Run enrichment on all .ttl files in a domain folder:

```bash
python -m src.pipeline \
  --domain <domain> \
  --input <path_to_nif_files> \
  --output <output_dir> \
  --step <stage> \
  [--batch_size N] \
  [--resume]
```

### ⚙️ Command-line options

| Flag                | Description                                                                                                                                                                                                                                                                                         |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--domain <domain>` | Domain name (e.g. `neuro`, `energy`, `ccam`, `maritime`, `cancer`). Used for logging and model selection.                                                                                                                                                                                           |
| `--input`           | Path to a directory containing `.ttl` NIF files to process.                                                                                                                                                                                                                                         |
| `--output`          | Path to output directory where results and checkpoints are saved.                                                                                                                                                                                                                                   |
| `--step`            | Which pipeline stage to run:<br>• `ner` — Named Entity Recognition (NER)<br>• `geotagging` — GeoNER + role classification<br>• `affiliations` — Extract and enrich affiliations using **AffilGood**<br>• `link` — (coming soon) entity linking<br>• `all` — (reserved for full multi-step pipeline) |
| `--batch_size`      | Number of `.ttl` files per batch (default: 1000). Smaller = lower memory usage.                                                                                                                                                                                                                     |
| `--resume`          | Resume from a saved checkpoint to skip previously processed files.                                                                                                                                                                                                                                  |
### ⚡ Examples

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

🧩 The pipeline automatically:

* Resumes from checkpoints (outputs/<domain>/checkpoints/processed.json)
* Logs all progress under outputs/<domain>/logs/
* Skips already-processed files
* Saves partial results safely in case of interruption

---
## 🧩 Entity Linking (NEL)

Each domain has its own KB:
| Domain   | Knowledge Base           | Notes                     |
| -------- | ------------------------ | ------------------------- |
| Neuro    | openMINDS, UBERON        | CNS-limited               |
| CCAM     | project-specific sheet   | To be expanded            |
| Energy   | IRENA                    | Renewable energy taxonomy |
| Maritime | project taxonomy         | provided by partners      |
| Cancer   | NCBI, DO, MeSH, Drugbank | biomedical baseline       |

Fallback: **Wikification** for unknown entities.

---
## 📦 NIF Format

Each enriched NIF file includes new triples like:
```ttl
<http://scilake.eu/resource#offset_42_58>
    a nif:EntityOccurrence ;
    nif:referenceContext <http://scilake.eu/resource#context_1> ;
    nif:beginIndex "42"^^xsd:int ;
    nif:endIndex "58"^^xsd:int ;
    nif:anchorOf "solar cell" ;
    itsrdf:taIdentRef <http://irena.org/kb/energy/SolarCell> .
```
---

## 📅 Next Steps

 - [x] Integrate acronym resolution
 - [x] Validate NEL KB coverage 
 - [ ] Deploy server for pilot-wide processing
 - [ ] Add evaluation & reporting
