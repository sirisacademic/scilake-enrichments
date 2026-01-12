#!/bin/bash

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --step ner \
    --input_format legal_text \
    --input data/fedlex-dataset-090425_translated.jsonl \
    --output outputs/energy-legal \
    --resume
