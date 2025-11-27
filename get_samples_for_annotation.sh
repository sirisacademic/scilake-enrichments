# Directory mode - sample files, then entities
python scripts/sample_extractor.py \
  --sections-dir outputs/ccam-all-ft/sections \
  --ner-dir outputs/ccam-all-ft/el \
  --taxonomy taxonomies/ccam/CCAM_Combined.tsv \
  --output outputs/ccam-all-ft/ccam_el_annotation_samples.tsv \
  --n-files 200 \
  --file-strategy model_balanced \
  --n-samples 500 \
  --strategy model_stratified \
  --stats

# Process all files, sample 500 entities
#python scripts/sample_extractor.py \
#  --sections-dir outputs/ccam-all-ft/sections \
#  --ner-dir outputs/ccam-all-ft/el \
#  --taxonomy taxonomies/ccam/CCAM_Combined.tsv \
#  --output outputs/ccam-all-ft/ccam_el_annotation_samples.tsv \
#  --n-files 0 \
#  --n-samples 500 \
#  --stats
