#!/bin/bash
# Run parallel EL for Energy domain on title/abstract JSON (6 parts)
# EL configuration is loaded from domain_models.py el_config

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

for i in 00 01 02 03 04 05; do
    echo "Starting EL part ${i}..."
    nohup python src/pipeline.py \
        --domain energy \
        --step el \
        --output outputs/title_abstract_json/energy/energy-titleabstract-part${i} \
        --resume \
        > outputs/title_abstract_json/energy/energy-titleabstract-part${i}_el.log 2>&1 &
done

echo "All 6 EL instances started."
echo "Monitor with: tail -f outputs/title_abstract_json/energy/energy-titleabstract-part*_el.log"
echo "Check GPU: nvidia-smi"
