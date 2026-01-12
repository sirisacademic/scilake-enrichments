#!/bin/bash
# =============================================================================
# Generate annotation samples for cancer domain - one file per entity type
# =============================================================================

set -e

SECTIONS_DIR="outputs/cancer-all-ft/sections"
NER_DIR="outputs/cancer-all-ft/el"
OUTPUT_DIR="outputs/cancer-all-ft"

N_FILES=1000
N_SAMPLES=500
FILE_STRATEGY="model_balanced"
STRATEGY="model_stratified"

echo "============================================================"
echo "Generating cancer annotation samples per entity type"
echo "============================================================"

# -----------------------------------------------------------------------------
# Gene samples
# -----------------------------------------------------------------------------
echo ""
echo ">>> Generating Gene samples..."
python scripts/sample_extractor.py \
  --sections-dir ${SECTIONS_DIR} \
  --ner-dir ${NER_DIR} \
  --output ${OUTPUT_DIR}/cancer_el_annotation_samples_GENE.tsv \
  --entity-type Gene \
  --n-files ${N_FILES} \
  --file-strategy ${FILE_STRATEGY} \
  --n-samples ${N_SAMPLES} \
  --strategy ${STRATEGY} \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

# -----------------------------------------------------------------------------
# Disease samples
# -----------------------------------------------------------------------------
echo ""
echo ">>> Generating Disease samples..."
python scripts/sample_extractor.py \
  --sections-dir ${SECTIONS_DIR} \
  --ner-dir ${NER_DIR} \
  --output ${OUTPUT_DIR}/cancer_el_annotation_samples_DISEASE.tsv \
  --entity-type Disease \
  --n-files ${N_FILES} \
  --file-strategy ${FILE_STRATEGY} \
  --n-samples ${N_SAMPLES} \
  --strategy ${STRATEGY} \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

# -----------------------------------------------------------------------------
# Species samples
# -----------------------------------------------------------------------------
echo ""
echo ">>> Generating Species samples..."
python scripts/sample_extractor.py \
  --sections-dir ${SECTIONS_DIR} \
  --ner-dir ${NER_DIR} \
  --output ${OUTPUT_DIR}/cancer_el_annotation_samples_SPECIES.tsv \
  --entity-type Species \
  --n-files ${N_FILES} \
  --file-strategy ${FILE_STRATEGY} \
  --n-samples ${N_SAMPLES} \
  --strategy ${STRATEGY} \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

# -----------------------------------------------------------------------------
# Chemical samples
# -----------------------------------------------------------------------------
echo ""
echo ">>> Generating Chemical samples..."
python scripts/sample_extractor.py \
  --sections-dir ${SECTIONS_DIR} \
  --ner-dir ${NER_DIR} \
  --output ${OUTPUT_DIR}/cancer_el_annotation_samples_CHEMICAL.tsv \
  --entity-type Chemical \
  --n-files ${N_FILES} \
  --file-strategy ${FILE_STRATEGY} \
  --n-samples ${N_SAMPLES} \
  --strategy ${STRATEGY} \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

# -----------------------------------------------------------------------------
# CellLine samples
# -----------------------------------------------------------------------------
echo ""
echo ">>> Generating CellLine samples..."
python scripts/sample_extractor.py \
  --sections-dir ${SECTIONS_DIR} \
  --ner-dir ${NER_DIR} \
  --output ${OUTPUT_DIR}/cancer_el_annotation_samples_CELLLINE.tsv \
  --entity-type CellLine \
  --n-files ${N_FILES} \
  --file-strategy ${FILE_STRATEGY} \
  --n-samples ${N_SAMPLES} \
  --strategy ${STRATEGY} \
  --include-unlinked \
  --unlinked-ratio 0.2 \
  --stats

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Done! Generated sample files:"
echo "============================================================"
ls -la ${OUTPUT_DIR}/cancer_el_annotation_samples_*.tsv
