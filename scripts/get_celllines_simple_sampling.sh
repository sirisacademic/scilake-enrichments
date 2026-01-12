python scripts/simple_sample_extractor.py \
  --sections-dir outputs/cancer-all-ft/sections \
  --ner-dir outputs/cancer-all-ft/el \
  --output outputs/cancer-all-ft/cancer_el_annotation_samples_CELLLINE.tsv \
  --entity-type CellLine \
  --n-linked 240 \
  --n-unlinked 60 \
  --max-per-mention 1 \
  --shuffle

