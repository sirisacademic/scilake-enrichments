#!/bin/bash
# Run full NER + EL pipeline for Neuro domain (NIF input)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain neuro \
    --input data/neuro-all-ft \
    --output outputs/neuro-all-ft \
    --step all \
    --resume
