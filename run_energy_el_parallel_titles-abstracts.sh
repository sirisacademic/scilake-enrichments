#!/bin/bash
# run_energy_el_parallel.sh

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

for i in 00 01 02 03 04 05; do
    echo "Starting EL part ${i}..."
    nohup python src/pipeline.py \
        --domain energy \
        --step el \
        --input_format title_abstract \
        --taxonomy taxonomies/energy/IRENA.tsv \
        --taxonomy_source IRENA \
        --output outputs/energy-titleabstract-part${i} \
        --linker_type reranker \
        --el_model_name intfloat/multilingual-e5-large-instruct \
        --threshold 0.70 \
        --context_window 5 \
        --max_contexts 5 \
        --reranker_llm Qwen/Qwen3-1.7B \
        --reranker_top_k 7 \
        --reranker_fallbacks \
        --resume \
        > outputs/energy-titleabstract-part${i}_el.log 2>&1 &
done

echo "All 6 EL instances started."
echo "Monitor with: tail -f outputs/energy-titleabstract-part*_el.log"
echo "Check GPU: nvidia-smi"

