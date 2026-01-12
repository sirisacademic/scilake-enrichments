#!/bin/bash
# =============================================================================
# Generate annotation samples for energy domain
# =============================================================================

# Directory mode - sample files, then entities
python scripts/sample_extractor.py \
  --sections-dir outputs/energy-all-ft/sections \
  --ner-dir outputs/energy-all-ft/el \
  --output outputs/energy-all-ft/energy_el_annotation_samples.tsv \
  --n-files 1000 \
  --file-strategy model_balanced \
  --n-samples 1000 \
  --strategy model_stratified \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats
