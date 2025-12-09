#!/bin/bash
export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

python -m src.pipeline --domain energy --input data/energy-all-ft --output outputs/energy-all-ft --step geotagging --batch_size 100

