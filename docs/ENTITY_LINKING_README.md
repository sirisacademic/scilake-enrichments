# Entity Linking Module - Documentation

## Overview

The Entity Linking (EL) module enriches NER-detected entities with semantic links to the IRENA taxonomy using multilingual-e5-base embeddings. It links entities based on their sentence context using asymmetric query/passage matching.

---

## Architecture

### Components

1. **SemanticLinker** (`src/semantic_linker.py`)
   - Loads and indexes IRENA taxonomy with embeddings
   - Extracts sentence context for entities
   - Performs semantic similarity matching
   - Manages linking cache

2. **run_el()** (`src/pipeline.py`)
   - Orchestrates entity linking workflow
   - Manages checkpoints and resume functionality
   - Processes NER outputs in batches
   - Saves linked outputs with statistics

---

## How It Works

### 1. IRENA Index Building (One-time)

```python
# Loads IRENA.tsv and creates in-memory index
passages = [
    "passage: Wind energy",      # Primary concept
    "passage: wind power",        # Alias 1
    "passage: wind power energy"  # Alias 2
]

embeddings = model.encode(passages)  # Shape: [N, 768]
```

**Index Structure:**
```python
{
    'embeddings': np.array([...]),  # [2000, 768]
    'metadata': [
        {
            'irena_id': '230000',
            'matched_text': 'Wind energy',
            'wikidata_id': 'Q43302',
            'type': 'Renewables'
        },
        ...
    ]
}
```

### 2. Entity Linking Process

For each NER entity without linking:

```python
# 1. Extract sentence containing first occurrence
sentence = "Wind turbines harness kinetic energy from wind."

# 2. Encode as query
query = "query: Wind turbines harness kinetic energy from wind."
query_emb = model.encode(query)

# 3. Compute similarities with all IRENA passages
scores = query_emb @ irena_index['embeddings'].T

# 4. Select best match if above threshold
best_match = irena_index['metadata'][argmax(scores)]
if scores.max() >= 0.6:
    linking = [
        {
            'source': 'IRENA',
            'id': '230000',
            'name': 'Wind energy',
            'score': 0.87
        },
        {
            'source': 'Wikidata',
            'id': 'Q43302',
            'name': 'Wind energy'
        }
    ]
```

### 3. Caching Strategy

**Cache Structure:**
```python
{
    "wind turbines": {
        "linking": [...],
        "sentence": "Wind turbines harness..."
    },
    "solar panels": {
        "linking": [...],
        "sentence": "Solar panels convert..."
    }
}
```

**Benefits:**
- Avoids re-computing same entity across documents
- High hit rate in domain-specific corpus
- Persistent across runs (saved to JSON)

---

## Usage

### Run Entity Linking Only

```bash
python src/pipeline.py \
    --domain energy \
    --output outputs/energy \
    --step el \
    --threshold 0.6 \
    --taxonomy IRENA.tsv \
    --resume
```

### Run Full Pipeline (NER + EL)

```bash
python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --step all \
    --threshold 0.6 \
    --batch_size 1000
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--domain` | Domain name (energy, ccam, etc.) | Required |
| `--output` | Output directory | Required |
| `--step` | Pipeline step: `ner`, `el`, or `all` | `ner` |
| `--threshold` | Minimum similarity score for linking | `0.6` |
| `--taxonomy` | Path to taxonomy TSV file | `taxonomies/energy/IRENA.tsv` |
| `--resume` | Resume from checkpoint | Flag |
| `--debug` | Enable debug logging | Flag |

---

## Input/Output Format

### Input (NER Output)

```json
{
  "section_id": "http://scilake.eu/resource#section_123",
  "entities": [
    {
      "entity": "energytype",
      "text": "wind turbines",
      "score": 0.92,
      "model": "SIRIS-Lab/SciLake-Energy-roberta-base",
      "domain": "energy",
      "section_id": "http://scilake.eu/resource#section_123"
    }
  ]
}
```

### Output (EL Output)

```json
{
  "section_id": "http://scilake.eu/resource#section_123",
  "entities": [
    {
      "entity": "energytype",
      "text": "wind turbines",
      "score": 0.92,
      "model": "SIRIS-Lab/SciLake-Energy-roberta-base",
      "domain": "energy",
      "section_id": "http://scilake.eu/resource#section_123",
      "linking": [
        {
          "source": "IRENA",
          "id": "230000",
          "name": "Wind energy",
          "score": 0.87
        },
        {
          "source": "Wikidata",
          "id": "Q43302",
          "name": "Wind energy"
        }
      ]
    }
  ]
}
```

---

## File Structure

```
outputs/energy/
├── ner/                      # NER outputs (input to EL)
│   ├── paper1.jsonl
│   ├── paper2.jsonl
│   └── expanded/             # Text context for linking
│       ├── paper1_expanded.csv
│       └── paper2_expanded.csv
│
└── el/                       # Entity Linking outputs
    ├── paper1.jsonl          # Entities WITH linking
    ├── paper2.jsonl
    ├── linking_cache.json    # Persistent cache
    ├── checkpoints/
    │   └── processed.json
    └── logs/
        └── energy_el.log
```

---

## Performance Considerations

### Memory Usage

- **IRENA embeddings**: ~6 MB (2000 entries × 768 dims × 4 bytes)
- **Sentence transformer model**: ~500 MB
- **Cache**: ~10 MB for 10,000 unique entities

**Total RAM requirement**: ~1 GB

### Speed

- **IRENA indexing**: ~100 seconds (one-time, at startup)
- **Per entity**:
  - Cache hit: < 1ms
  - Cache miss: ~10ms (sentence extraction + encoding + matching)
  
- **Expected throughput**:
  - With 80% cache hit rate: ~500 entities/second
  - Cold start (0% cache): ~100 entities/second

### Cache Hit Rate Evolution

```
Documents processed:    0    100   500   1000  5000
Cache hit rate:        0%    40%   70%    80%   90%
```

---

## Tuning Parameters

### Threshold Selection

The `threshold` parameter controls linking precision/recall tradeoff:

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.5 | High recall, lower precision | Exploratory analysis |
| 0.6 | **Balanced** (recommended) | General use |
| 0.7 | High precision, lower recall | High-quality annotations |
| 0.8+ | Very strict | Manual review required |

**Recommendation**: Start with 0.6 and adjust based on manual evaluation.

### Sentence Context

Currently uses full sentence containing entity. Alternatives:

- **Wider context**: Extract paragraph (may dilute signal)
- **Narrower context**: ±50 chars window (faster, less context)
- **Multiple sentences**: Current + adjacent sentences

---

## Troubleshooting

### Issue: Low linking rate (<30%)

**Possible causes:**
1. Threshold too high → Lower to 0.5
2. Domain mismatch → Check if taxonomy covers domain
3. Poor sentence segmentation → Check spaCy model

**Debug:**
```bash
python src/pipeline.py --step el --debug
# Check logs/energy_el.log for entity-level scores
```

### Issue: Out of memory

**Solutions:**
1. Process fewer files per batch
2. Clear cache periodically
3. Use CPU instead of GPU for embeddings

### Issue: Slow performance

**Check:**
1. Cache hit rate (should be >70% after 100 docs)
2. GPU availability (10x faster)
3. Network access (if downloading models)

---

## Testing

### Unit Test

```bash
bash /mnt/user-data/outputs/test_semantic_linker.sh
```

Tests:
1. IRENA index loading
2. Sentence extraction
3. Basic entity linking
4. Multiple entities in section
5. Cache functionality

### Integration Test

```bash
# Run on small sample
python src/pipeline.py \
    --domain energy \
    --input data/energy_sample \
    --output outputs/test \
    --step all \
    --batch_size 10
```

---

## Future Enhancements

### Planned Features

1. **Multi-taxonomy support**: Link to multiple KBs simultaneously
2. **Confidence calibration**: Improve score interpretation
3. **Disambiguation**: Handle polysemous entities
4. **Active learning**: Flag uncertain links for review
5. **Cross-lingual**: Support non-English documents

### Optimization Opportunities

1. **FAISS indexing**: Faster similarity search (100x speedup)
2. **Batch encoding**: Process multiple entities together
3. **Approximate search**: Trade accuracy for speed
4. **Quantization**: Reduce memory footprint

---

## API Reference

### SemanticLinker

```python
class SemanticLinker:
    def __init__(
        self,
        taxonomy_path: str,
        model_name: str = "intfloat/multilingual-e5-base",
        threshold: float = 0.6,
        logger = None
    )
    
    def link_entity(
        self,
        entity_text: str,
        sentence: str
    ) -> Optional[List[Dict]]
    
    def link_entities_in_section(
        self,
        section_text: str,
        entities: List[Dict],
        cache: Dict
    ) -> Tuple[List[Dict], Dict]
```

### run_el()

```python
def run_el(
    domain: str,
    ner_output_dir: str,
    el_output_dir: str,
    taxonomy_path: str = "taxonomies/energy/IRENA.tsv",
    threshold: float = 0.6,
    resume: bool = True,
    debug: bool = False
)
```

---

## References

- **E5 Model**: [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)
- **IRENA Taxonomy**: Energy-specific ontology from IRENA
- **NIF Format**: [NLP Interchange Format](http://persistence.uni-leipzig.org/nlp2rdf/)

---

## Support

For issues or questions:
1. Check logs: `outputs/{domain}/el/logs/`
2. Enable debug mode: `--debug`
3. Review cache statistics in logs
4. Test with sample data first
