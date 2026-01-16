#!/bin/bash
# Run full NER + EL pipeline for Maritime domain (NIF input)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain maritime \
    --input data/maritime-all-ft \
    --output outputs/maritime-all-ft \
    --step all \
    --resume
