# Directory mode - sample files, then entities
python scripts/sample_extractor.py \
  --sections-dir outputs/neuro-all-ft/sections \
  --ner-dir outputs/neuro-all-ft/el \
  --taxonomy taxonomies/neuro/Neuroscience_Combined.tsv \
  --output outputs/neuro-all-ft/neuro_el_annotation_samples.tsv \
  --n-files 1000 \
  --file-strategy model_balanced \
  --n-samples 1000 \
  --strategy model_stratified \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

