#!/bin/bash
# Run EL pipeline for Energy domain (NIF input)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --input data/energy-all-ft \
    --output outputs/energy-all-ft \
    --step el \
    --resume
