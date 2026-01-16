#!/bin/bash
# Run EL for Energy domain on legal text (Fedlex)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --step el \
    --output outputs/title_abstract_json/energy/energy-legal \
    --resume
