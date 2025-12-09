python scripts/merge_gazetteer_ner_el.py \
	--domain energy \
	--old-ner-dir outputs/energy-all-ft/ner.old \
	--new-gaz-dir outputs/energy-all-ft-gaz/ner \
	--output-dir outputs/energy-all-ft/ner
	
python scripts/merge_gazetteer_ner_el.py \
	--domain energy \
	--old-ner-dir outputs/energy-all-ft/el.old \
	--new-gaz-dir outputs/energy-all-ft-gaz/ner \
	--output-dir outputs/energy-all-ft/el
	


