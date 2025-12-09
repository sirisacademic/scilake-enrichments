#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# !!! Run with --resume to resume from previous run !!!

python src/pipeline.py \
    --domain ccam \
    --taxonomy taxonomies/ccam/CCAM_Combined.tsv \
    --taxonomy_source SINFONICA-FAME \
    --input data/ccam-all-ft \
    --output outputs/ccam-all-ft \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --context_window 5 \
    --max_contexts 5 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks

