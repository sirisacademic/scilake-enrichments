#!/bin/bash
# run_cancer_ner_parallel.sh

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

for i in 00 01 02 03 04 05; do
    echo "Starting part ${i}..."
    nohup python src/pipeline.py \
        --domain cancer \
        --step ner \
        --input_format title_abstract \
        --input data/title_abstract_json/cancer/cancer_titleabstract_part${i}.json \
        --output outputs/title_abstract_json/cancer/cancer-titleabstract-part${i} \
        --resume \
        > outputs/title_abstract_json/cancer/cancer-titleabstract-part${i}.log 2>&1 &
done

echo "All 6 instances started."
echo "Monitor with: tail -f outputs/title_abstract_json/cancer/cancer-titleabstract-part*.log"
echo "Check GPU: nvidia-smi"
echo "List jobs: ps aux | grep pipeline"
