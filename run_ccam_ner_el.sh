#!/bin/bash
# Run full NER + EL pipeline for CCAM domain (NIF input)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain ccam \
    --input data/ccam-all-ft \
    --output outputs/ccam-all-ft \
    --step all \
    --resume
