#!/bin/bash
# run_energy_ner_parallel.sh

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

for i in 00 01 02 03 04 05; do
    echo "Starting part ${i}..."
    nohup python src/pipeline.py \
        --domain energy \
        --step ner \
        --input_format title_abstract \
        --input /root/scilake-enrichments/data/title_abstract_json/energy_titleabstract_part${i}.json \
        --output outputs/energy-titleabstract-part${i} \
        --resume \
        > outputs/energy-titleabstract-part${i}.log 2>&1 &
done

echo "All 6 instances started."
echo "Monitor with: tail -f outputs/energy-titleabstract-part*.log"
echo "Check GPU: nvidia-smi"
echo "List jobs: ps aux | grep pipeline"
