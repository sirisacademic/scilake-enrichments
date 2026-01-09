#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"
    
python src/pipeline.py \
    --domain energy \
    --step all \
    --input_format title_abstract \
    --taxonomy taxonomies/energy/IRENA.tsv \
    --taxonomy_source IRENA \
    --input /root/scilake-enrichments/data/title_abstract_json/sample-energy_titleabstract.json \
    --output outputs/sample-energy-titleabstract \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --context_window 5 \
    --max_contexts 5 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks \
    --resume

