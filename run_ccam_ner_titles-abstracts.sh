#!/bin/bash

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "Starting NER for CCAM Titles/Abstracts..."

nohup python src/pipeline.py \
	--domain ccam \
	--step ner \
	--input_format title_abstract \
	--input data/title_abstract_json/ccam/ccam_titleabstract.json \
	--output outputs/title_abstract_json/ccam/ccam-titleabstract \
	--resume \
	> outputs/title_abstract_json/ccam/ccam-titleabstract.log 2>&1 &
	
echo "Monitor with: tail -f outputs/title_abstract_json/ccam/ccam-titleabstract.log"


