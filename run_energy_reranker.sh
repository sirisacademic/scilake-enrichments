#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --input data/energy-all-ft \
    --output outputs/energy-all-ft \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --context_window 5 \
    --max_contexts 5 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks

