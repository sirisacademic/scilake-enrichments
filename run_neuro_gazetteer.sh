#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# !!! Run with --resume to resume from previous run !!!

python src/pipeline.py \
    --step gaz \
    --domain neuro \
    --taxonomy taxonomies/neuro/Neuroscience_Combined.tsv \
    --taxonomy_source OPENMINDS-UBERON \
    --input data/neuro-all-ft \
    --output outputs/neuro-all-ft-gaz \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --context_window 5 \
    --max_contexts 5 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks \
    --resume

