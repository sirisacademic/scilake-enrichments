#!/bin/bash
# Run EL for Maritime domain on title/abstract JSON
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "Starting EL for Maritime Titles/Abstracts..."

nohup python src/pipeline.py \
    --domain maritime \
    --step el \
    --output outputs/title_abstract_json/maritime/maritime-titleabstract \
    --resume \
    > outputs/title_abstract_json/maritime/maritime-titleabstract_el.log 2>&1 &

echo "Monitor with: tail -f outputs/title_abstract_json/maritime/maritime-titleabstract_el.log"
