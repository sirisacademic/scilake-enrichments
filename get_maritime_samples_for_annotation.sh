# Directory mode - sample files, then entities
python scripts/sample_extractor.py \
  --sections-dir outputs/maritime-all-ft/sections \
  --ner-dir outputs/maritime-all-ft/el \
  --taxonomy taxonomies/maritime/VesselTypes.tsv \
  --output outputs/maritime-all-ft/maritime_el_annotation_samples.tsv \
  --n-files 1000 \
  --file-strategy model_balanced \
  --n-samples 1000 \
  --strategy model_stratified \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

