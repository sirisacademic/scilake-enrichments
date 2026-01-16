#!/bin/bash

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# Create a test file with ~1000 records
head -n 1000 data/title_abstract_json/cancer/cancer_titleabstract_part00.json > data/title_abstract_json/cancer/test_sample.json

# Run NER
python -m src.pipeline \
    --domain cancer \
    --step ner \
    --input_format title_abstract \
    --input data/title_abstract_json/cancer/test_sample.json \
    --output outputs/cancer/test \
    --batch_size 100

# Run EL
python -m src.pipeline \
    --domain cancer \
    --step el \
    --input_format title_abstract \
    --input data/title_abstract_json/cancer/test_sample.json \
    --output outputs/cancer/test

# Check results
head outputs/cancer/test/el/*.jsonl | python -m json.tool


