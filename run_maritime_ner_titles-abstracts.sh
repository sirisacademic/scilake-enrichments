#!/bin/bash

export PYTHONPATH="$(dirname "$0"):$PYTHONPATH"

echo "Starting NER for Maritime Titles/Abstracts..."

nohup python src/pipeline.py \
	--domain maritime \
	--step ner \
	--input_format title_abstract \
	--input data/title_abstract_json/maritime/maritime_titleabstract.json \
	--output outputs/title_abstract_json/maritime/maritime-titleabstract \
	--resume \
	> outputs/title_abstract_json/maritime/maritime-titleabstract.log 2>&1 &
	
echo "Monitor with: tail -f outputs/title_abstract_json/maritime/maritime-titleabstract.log"


