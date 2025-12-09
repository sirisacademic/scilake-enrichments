"""
Domain-specific model configuration for SciLake NER/NEL enrichment.

IMPORTANT: 
- For GLiNER models: `labels` is REQUIRED - passed to model.run()
- For RoBERTa models: `labels` is DOCUMENTATION ONLY - model has fixed output labels

Each domain defines:
 - gazetteer: configuration for exact-match gazetteer linking
 - models: list of NER models to apply (GLiNER / RoBERTa)
 - labels: label sets (required for GLiNER, informational for RoBERTa)
 - kb: knowledge base(s) to use for Entity Linking
 - linking_strategy: "semantic" (default), "reranker", or "fts5"
 - fts5_linkers: (if linking_strategy="fts5") per-entity-type FTS5 index config
"""

DOMAIN_MODELS = {
    "neuro": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/neuro/Neuroscience_Combined.tsv",
            "taxonomy_source": "OPENMINDS-UBERON",
            "model_name": "Neuroscience-Gazetteer",
            "default_type": "UBERONParcellation"
        },
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Neuroscience-GLiNER-large",
                "type": "gliner",
                "threshold": 0.95,
                # GLiNER labels - REQUIRED, passed to model.run()
                "labels": [
                    "UBERONParcellation",
                    "species",
                    "preparationType",
                    "technique",
                    "biologicalSex",
                ],
            },
        ],
        # Domain-level labels for GLiNER (alternative lookup)
        "labels": {
            "gliner": [
                "UBERONParcellation",
                "species",
                "preparationType",
                "technique",
                "biologicalSex",
            ],
        },
        "kb": {
            "UBERONParcellation": "UBERON (restricted to CNS)",
            "others": "openMINDS controlled terms",
        },
    },
    
    "ccam": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/ccam/CCAM_Combined.tsv",
            "taxonomy_source": "SINFONICA-FAME",
            "model_name": "CCAM-Gazetteer",
            "default_type": "General terms related to CCAM",
            "min_term_length": 2,
            "blocked_terms": {
                # Too generic - appear in any technical text
                'data', 'user', 'system', 'case', 'model', 'action', 'event',
                'process', 'range', 'message', 'service', 'function', 'information',
                'location', 'test', 'entity', 'scene', 'situation',
                'failure', 'unsuccessful', 'danger', 'risk', 'risks',
                'sign', 'indication', 'route', 'pitch', 'slope',
                'target', 'audience', 'message', 'communication',
                'place', 'position', 'site', 'location',
                'auto', 'statement', 'theme',
                # Ambiguous without CCAM context
                'actor', 'monitor', 'treatment', 'trial', 'pilot', 'exposure',
                'trigger', 'baseline', 'indicator', 'component',
                # Common words that match taxonomy but cause noise
                'standard', 'authorization', 'consent', 'privacy', 'interface',
                'assessment', 'evaluation', 'validation', 'verification',
            }
        },
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-CCAM-roberta-large-vehicle-vru",
                "type": "roberta",
                "threshold": 0.7,
                # RoBERTa labels - DOCUMENTATION ONLY (model outputs these)
                "output_labels": ["vehicleType", "VRUType"],
            },
            {
                "name": "SIRIS-Lab/SciLake-CCAM-roberta-large-other",
                "type": "roberta",
                "threshold": 0.7,
                # RoBERTa labels - DOCUMENTATION ONLY (model outputs these)
                "output_labels": [
                    "scenarioType",
                    "communicationType",
                    "entityConnectionType",
                    "levelOfAutomation",
                    "sensorType"
                ],
            }
        ],
        # Labels dict - not used for RoBERTa, kept for reference
        "labels": {
            "roberta": [
                "vehicleType", "VRUType",
                "scenarioType", "communicationType", 
                "entityConnectionType", "levelOfAutomation", "sensorType"
            ],
        },
        "kb": {
            "default": "Project-specific CCAM vocabulary (pilot sheet)",
        },
    },
    
    "energy": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "taxonomy_source": "IRENA",
            "model_name": "IRENA-Gazetteer",
            "default_type": "EnergyType"
        },
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Energy-roberta-base",
                "type": "roberta",
                "threshold": 0.85,
                # RoBERTa labels - DOCUMENTATION ONLY
                "output_labels": ["EnergyType", "EnergyStorage"],
            },
        ],
        "labels": {
            "roberta": ["EnergyType", "EnergyStorage"],
        },
        "kb": {
            "default": "IRENA energy taxonomy",
        },
    },
    
    "maritime": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/maritime/VesselTypes.tsv",
            "taxonomy_source": "Maritime-Ontology",
            "model_name": "Maritime-Gazetteer",
            "default_type": "vesselType"
        },
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Maritime-roberta-base",
                "type": "roberta",
                "threshold": 0.75,
                # RoBERTa labels - DOCUMENTATION ONLY
                "output_labels": ["vesselType"],
            },
        ],
        "labels": {
            "roberta": ["vesselType"],
        },
        "kb": {
            "default": "Maritime ontology provided by partners",
        },
    },
    
    "cancer": {
        # Disable combined gazetteer - use FTS5 per entity type instead
        "gazetteer": {
            "enabled": False,
        },
        "models": [
            {
                "name": "SIRIS-Lab/AIObioEnts-core-pubmedbert-full",
                "type": "aioner",
                "threshold": 0.5,
            },
        ],
        "labels": {
            "aioner": [
                "Gene",
                "Disease",
                "CellLine",
                "Chemical",
                "Species",
                "Variant",
            ],
        },
        "kb": {
            "Gene": "NCBI Gene",
            "Disease": "Disease Ontology",
            "CellLine": "BRENDA",
            "Chemical": "DrugBank",
            "Species": "NCBI Taxonomy",
        },
        
        # New: FTS5-based linking strategy
        "linking_strategy": "fts5",
        
        # FTS5 linker configuration per entity type
        "fts5_linkers": {
            "gene": {
                "index_path": "indices/cancer/ncbi_gene.db",
                "taxonomy_source": "NCBI_Gene",
                "taxonomy_path": "taxonomies/cancer/NCBI_GENE.tsv",  # For reference
            },
            "species": {
                "index_path": "indices/cancer/ncbi_species.db",
                "taxonomy_source": "NCBI_Taxonomy",
                "taxonomy_path": "taxonomies/cancer/NCBI_SPECIES.tsv",
                "blocked_mentions": {
                    # === Original blocked terms ===
                    "patient", "patients", 
                    "man", "men", "woman", "women",
                    "inpatient", "outpatient",
                    # Patient/person variants
                    "human patients", "human patient", "patienten",  # German
                    "people", "person", "persons",
                    "inpatients", "outpatients",
                    # Medical personnel
                    "nurse", "nurses", "doctor", "doctors",
                    # Gender terms
                    "female", "male", "females", "males",
                    # Lifestyle/condition terms  
                    "smoker", "smokers", "plwh", "plhiv",  # people living with HIV
                    # Microorganisms (too generic or not species-specific)
                    "bacteria", "bacterial", "virus",
                    "fungi", "fungal", "mushroom", "mushrooms",
                    "algae", "microalgae",
                    # Plants
                    "plant",
                    # Colors (false positives from species names)
                    "green", "red", "black",
                    # Single letters and numbers (noise)
                    "a", "b", "c", "e", "h", "m", "s",
                    "1", "2",
                    # Other noise/false positives
                    "ad", "nod", "gas", "waterpipe", "spions",
                    # Punctuation artifacts
                    "-", ".", "- 19",
                },
            },
            "disease": {
                "index_path": "indices/cancer/doid_disease.db",
                "taxonomy_source": "DOID",
                "taxonomy_path": "taxonomies/cancer/DOID_DISEASE.tsv",
                "blocked_mentions": {
                    # === Biological processes / mechanisms (not diseases) ===
                    "inflammatory", "inflammation", "chronic inflammation", "proinflammatory",
                    "toxicity", "toxicities", "cytotoxicity", "cytotoxic",
                    "cardiotoxicity", "hepatotoxicity", "nephrotoxicity",
                    "ferroptosis", "pyroptosis", "cuproptosis",
                    "hypoxia", "hypoxic",
                    "necrosis", "necrotic",
                    "neurodegeneration", "neuroinflammation",
                    # === Symptoms / clinical signs ===
                    "death", "deaths",
                    "pain", "abdominal pain", "chronic pain", "neuropathic pain",
                    "fatigue", "frailty",
                    "bleeding", "blood loss", "hemorrhage",
                    "nausea", "vomiting", "nausea and vomiting",
                    "fever", "hyperthermia",
                    "cough", "dyspnea",
                    "headache",
                    "insomnia",
                    "constipation",
                    "dysphagia",
                    "edema", "swelling",
                    "rash",
                    "hearing loss",
                    "weight loss", "weight gain",
                    "psychological distress", "distress",
                    # === Clinical complications / outcomes ===
                    "infection", "infections",
                    "sepsis",
                    "postoperative complications", "complications",
                    "acute kidney injury",
                    "liver injury",
                    "fracture", "fractures",
                    "trauma",
                    "wound",
                    "perforation",
                    "fistula",
                    # === Generic clinical terms ===
                    "malnutrition",
                    "cachexia",
                    "overweight",
                    "smoking", "tobacco",
                    # === Anatomical / pathological features (not diseases) ===
                    "ascites",
                    "pleural effusion", "malignant pleural effusion",
                    "polyp", "polyps",
                    "lymph node",
                    "thyroid nodules",
                    "skin lesions",
                    "precancerous lesions",
                    "aneuploidy",
                    # === Specific cancer models (not human diseases) ===
                    "lewis lung carcinoma",
                    # === Treatment-related terms ===
                    "oral mucositis",
                    "xerostomia",
                    "iraes", "irae",  # immune-related adverse events
                    # === Miscellaneous ===
                    "tumor immune dysfunction",
                    "ncds",  # non-communicable diseases (too generic)
                    "ov",    # likely abbreviation noise
                    "breast",  # anatomical term, not disease
                },
            },
            "chemical": {
                "index_path": "indices/cancer/drugbank_chemical.db",
                "taxonomy_source": "DrugBank",
                "taxonomy_path": "taxonomies/cancer/DRUGBANK_CHEMICAL.tsv",
            },
            "cellline": {
                "index_path": "indices/cancer/brenda_cellline.db",
                "taxonomy_source": "BRENDA",
                "taxonomy_path": "taxonomies/cancer/BRENDA_CELLLINE.tsv",
            },
            # Variant: skip for now (no vocabulary)
        },
    },
}
