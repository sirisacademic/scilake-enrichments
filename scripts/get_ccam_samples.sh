#!/bin/bash
# =============================================================================
# Generate annotation samples for ccam domain
# =============================================================================

# Directory mode - sample files, then entities
python scripts/sample_extractor.py \
  --sections-dir outputs/ccam-all-ft/sections \
  --ner-dir outputs/ccam-all-ft/el \
  --output outputs/ccam-all-ft/ccam_el_annotation_samples.tsv \
  --n-files 1000 \
  --file-strategy model_balanced \
  --n-samples 1000 \
  --strategy model_stratified \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats
