python enrich_annotation_samples.py \
  --samples /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_GENE.tsv \
  --taxonomy /root/scilake-enrichments/taxonomies/cancer/NCBI_GENE.tsv \
  --el-dir /root/scilake-enrichments/outputs/cancer-all-ft/el \
  --output /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_GENE_enriched.tsv

python enrich_annotation_samples.py \
  --samples /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_DISEASE.tsv \
  --taxonomy /root/scilake-enrichments/taxonomies/cancer/DOID_DISEASE.tsv \
  --el-dir /root/scilake-enrichments/outputs/cancer-all-ft/el \
  --output /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_DISEASE_enriched.tsv
  
  python enrich_annotation_samples.py \
  --samples /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_SPECIES.tsv \
  --taxonomy /root/scilake-enrichments/taxonomies/cancer/NCBI_SPECIES.tsv \
  --el-dir /root/scilake-enrichments/outputs/cancer-all-ft/el \
  --output /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_SPECIES_enriched.tsv
  
  python enrich_annotation_samples.py \
  --samples /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_CHEMICAL.tsv \
  --taxonomy /root/scilake-enrichments/taxonomies/cancer/DRUGBANK_CHEMICAL.tsv \
  --el-dir /root/scilake-enrichments/outputs/cancer-all-ft/el \
  --output /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_CHEMICAL_enriched.tsv
  
  python enrich_annotation_samples.py \
  --samples /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_CELLLINE.tsv \
  --taxonomy /root/scilake-enrichments/taxonomies/cancer/BRENDA_CELLLINE.tsv \
  --el-dir /root/scilake-enrichments/outputs/cancer-all-ft/el \
  --output /root/scilake-enrichments/outputs/cancer-all-ft/cancer_el_annotation_samples_CELLLINE_enriched.tsv
