#!/bin/bash
# Run full NER + EL pipeline for Cancer domain (NIF input)
# Uses FTS5 linking (automatic from domain config)

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain cancer \
    --input data/cancer-all-ft \
    --output outputs/cancer-all-ft \
    --step all \
    --resume
