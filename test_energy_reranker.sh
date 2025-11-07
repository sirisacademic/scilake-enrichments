#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python src/pipeline.py \
    --domain energy \
    --input data/energy \
    --output outputs/energy \
    --linker_type reranker \
    --el_model_name intfloat/multilingual-e5-large-instruct \
    --threshold 0.70 \
    --context_window 5 \
    --max_contexts 5 \
    --reranker_llm Qwen/Qwen3-1.7B \
    --reranker_top_k 7 \
    --reranker_fallbacks

#    --reranker_llm Qwen/Qwen3-4B-Instruct-2507
#    --reranker_llm qingy2024/Benchmaxx-Llama-3.2-1B-Instruct
#    --reranker_llm Qwen/Qwen3-1.7B \
#    --reranker_llm microsoft/Phi-4-mini-instruct \
    
# Optional flags (uncomment to enable):
#    --use_context_for_retrieval \
#    --use_sentence_context \
#    --reranker_thinking


