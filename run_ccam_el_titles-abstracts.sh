#!/bin/bash
# Run EL for CCAM domain on title/abstract JSON
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "Starting EL for CCAM Titles/Abstracts..."

nohup python src/pipeline.py \
    --domain ccam \
    --step el \
    --output outputs/title_abstract_json/ccam/ccam-titleabstract \
    --resume \
    > outputs/title_abstract_json/ccam/ccam-titleabstract_el.log 2>&1 &

echo "Monitor with: tail -f outputs/title_abstract_json/ccam/ccam-titleabstract_el.log"
