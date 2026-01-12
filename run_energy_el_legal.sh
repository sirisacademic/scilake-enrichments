#!/bin/bash

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --step el \
    --input_format legal_text \
    --taxonomy taxonomies/energy/IRENA.tsv \
    --taxonomy_source IRENA \
    --output outputs/energy-legal \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks \
    --resume

