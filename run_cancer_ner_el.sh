#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

# !!! Run with --resume to resume from previous run !!!

# !!! Run without --step to run NER+EL !!!

python src/pipeline.py \
    --domain cancer \
    --step el \
    --input data/cancer-all-ft \
    --output outputs/cancer-all-ft \
    --resume
    

