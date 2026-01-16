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
 - el_config: entity linking configuration (threshold, context, model, etc.)
 - enforce_type_match: enable/disable type matching validation (default: True)
 - type_mappings: maps taxonomy type → expected NER type(s)
 - taxonomy_type_column: column name for type in taxonomy TSV (default: "type")

Entity Linking Configuration (el_config):
 - taxonomy_path: path to taxonomy TSV file
 - taxonomy_source: name for taxonomy source in output
 - linker_type: "semantic" | "instruct" | "reranker" | "fts5"
 - el_model_name: embedding model for semantic similarity
 - threshold: similarity threshold (0.0-1.0), default 0.80
 - context_window: token window for context extraction
 - max_contexts: maximum number of contexts per entity
 - use_sentence_context: use full sentences instead of token windows
 - reranker_llm: LLM model for reranker
 - reranker_top_k: number of candidates for reranker
 - reranker_fallbacks: add top-level fallback categories
"""

DOMAIN_MODELS = {

    #### NEURO ####
    "neuro": {
        # Gazetteer settings
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/neuro/Neuroscience_Combined.tsv",
            "taxonomy_source": "OPENMINDS-UBERON",
            "model_name": "Neuroscience-Gazetteer",
            "default_type": "UBERONParcellation"
        },
        # Set of ignored mentions that can be artifacts of the NER models.
        # This can be defined in general or at entity-type level (see example in cancer domain)
        "blocked_mentions": {},
        # Model configurations
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
        # Min. length of mentions extracted by NER models.
        # This could be set at entity-type level, too - see example in "cancer".
        "min_mention_length": 2,
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
        
        # === Entity Linking Configuration ===
        "linking_strategy": "reranker",
        "el_config": {
            "taxonomy_path": "taxonomies/neuro/Neuroscience_Combined.tsv",
            "taxonomy_source": "OPENMINDS-UBERON",
            "linker_type": "reranker",
            "el_model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.80,
            "context_window": 5,
            "max_contexts": 5,
            "use_sentence_context": False,
            "reranker_llm": "Qwen/Qwen3-1.7B",
            "reranker_top_k": 7,
            "reranker_fallbacks": True,
        },
        
        # === Type Matching Configuration ===
        "enforce_type_match": True,
        "taxonomy_type_column": "type",  # Column name in taxonomy TSV
        "type_mappings": {
            # Maps taxonomy type → expected NER type(s)
            # Format: "taxonomy_type": "ner_type" or ["ner_type1", "ner_type2"]
            "UBERONParcellation": "uberonparcellation",
            "technique": "technique",
            "species": "species",
            "preparationType": "preparationtype",
            "biologicalSex": "biologicalsex",
            # Brain regions may have different taxonomy types
            "brain_region": "uberonparcellation",
            "anatomical_structure": "uberonparcellation",
        },
    },
    
    ##### CCAM ####
    "ccam": {
        # Gazetteer settings
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
        # Min. length of mentions extracted by NER models.
        # This could be set at entity-type level, too - see example in "cancer".
        "min_mention_length": 2,       
        # Set of ignored mentions that can be artifacts of the NER models.
        # This can be defined in general or at entity-type level (see example in cancer domain)
        "blocked_mentions": {},
        # Model configurations
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
        
        # === Entity Linking Configuration ===
        "linking_strategy": "reranker",
        "el_config": {
            "taxonomy_path": "taxonomies/ccam/CCAM_Combined.tsv",
            "taxonomy_source": "SINFONICA-FAME",
            "linker_type": "reranker",
            "el_model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.80,
            "context_window": 5,
            "max_contexts": 5,
            "use_sentence_context": False,
            "reranker_llm": "Qwen/Qwen3-1.7B",
            "reranker_top_k": 7,
            "reranker_fallbacks": True,
        },
        
        # === Type Matching Configuration ===
        "enforce_type_match": True,
        "taxonomy_type_column": "entity_category",  # CCAM uses entity_category column
        "type_mappings": {
            # Maps taxonomy entity_category → expected NER type(s)
            "automation technologies": "levelofautomation",
            "communication types": "communicationtype",
            "entity connection types": "entityconnectiontype",
            "scenario types": "scenariotype",
            "sensor types": "sensortype",
            "vehicle types": "vehicletype",
            "VRU types": "vrutype",
            # Alternative category names that may appear
            "vehicle": "vehicletype",
            "vru": "vrutype",
            "sensor": "sensortype",
            "automation": "levelofautomation",
            "communication": "communicationtype",
            "scenario": "scenariotype",
        },
    },

    #### ENERGY ####
    "energy": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "taxonomy_source": "IRENA",
            "model_name": "IRENA-Gazetteer",
            "default_type": "EnergyType"
        },
        # Min. length of mentions extracted by NER models.
        # This could be set at entity-type level, too - see example in "cancer".
        "min_mention_length": 2,
        # Set of ignored mentions that can be artifacts of the NER models.
        # This can be defined in general or at entity-type level (see example in cancer domain)
        "blocked_mentions": {
            "aerosol", "aerosols", "agribusiness", "agrochemicals", "air preheater", "alkali", "alkaline", "alkaline sorbent", "ammonia borane", "ammonium",
            "anthropogenic aerosols", "asphalt", "basalt", "biofertilizers", "biogenic", "biogenic co 2", "biopolymers", "bivalve", "bromine", "cadmium",
            "calcium aluminate", "califonria", "candle lighting", "carbon", "carbon dioxide", "cascade heat exchanger", "cellulose nanomaterials", "co",
            "coals", "cobalt", "cocoa beans", "cocoa butter", "cracked", "cracked micro beams", "crassulacean acid", "date palm", "dihydrogen",
            "dish reflector", "dri-eaf", "dso", "electric arc furnace slag", "electric boilers", "electric heaters", "electrified",
            "electrochemical ammonia", "electrolytic manganese slag", "energy", "exchangers", "fertilizer", "fin heat exchangers", "fluorinated",
            "fly ash cement paste", "gas boiler", "gas-carbon", "gas ccs", "greenhouse", "greenhouse gas", "ground", "heat exchangers", "hybrid",
            "hydrochlorofluorocarbons", "internal", "internal combustion engine", "internal combustion engine cars", "iron ore", "isopropanol",
            "jet engines", "karst aquifers", "lithium polysulfide", "manganese", "manganese slag", "manure", "methanotrophs", "molten salt",
            "monocyclic", "nadph", "nanoparticles", "nh3", "nitric", "nitrification", "nitrite", "nox", "nuclides", "particulate", "pfass",
            "pha", "phosphogypsum", "phosphoric", "phosphoric acid", "phosphorus", "pin fin heat exchanger", "pin heat exchanger", "potassium", 
            "ppo", "proton", "pyrolytic", "pyruvate", "rap", "rare", "reheater", "sand", "sdfb", "selenoproteins", "slag", "so2", "sodium", "soot",
            "soot aerosol", "soybean", "soybeans", "ste", "steel", "sulfate", "syngas-hydrogen", "synthesis gas", "thermal power", "thiosulfate",
            "tsos", "ups", "volcanic", "west texas intermediate"
        },
        # Model configurations
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
        
        # === Entity Linking Configuration ===
        "linking_strategy": "reranker",
        "el_config": {
            "taxonomy_path": "taxonomies/energy/IRENA.tsv",
            "taxonomy_source": "IRENA",
            "linker_type": "reranker",
            "el_model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.80,
            "context_window": 5,
            "max_contexts": 5,
            "use_sentence_context": False,
            "reranker_llm": "Qwen/Qwen3-1.7B",
            "reranker_top_k": 7,
            "reranker_fallbacks": True,
        },
        
        # === Type Matching Configuration ===
        "enforce_type_match": True,
        "taxonomy_type_column": "type",
        "type_mappings": {
            # Maps taxonomy type → expected NER type(s)
            # Energy taxonomy types map to NER labels
            "Storage": "energystorage",
            "Renewables": "energytype",
            "Non-renewable": "energytype",
            "Total energy": "energytype",
            # Some taxonomies may have more specific types
            "Solar": "energytype",
            "Wind": "energytype",
            "Hydro": "energytype",
            "Biomass": "energytype",
            "Geothermal": "energytype",
            "Nuclear": "energytype",
            "Fossil": "energytype",
            "Battery": "energystorage",
            "Hydrogen": ["energytype", "energystorage"],  # Can be both
            # Fallback for generic energy types
            "Energy": ["energytype", "energystorage"],
        },
    },

    #### MARITIME ####
    "maritime": {
        "gazetteer": {
            "enabled": True,
            "taxonomy_path": "taxonomies/maritime/VesselTypes.tsv",
            "taxonomy_source": "Maritime-Ontology",
            "model_name": "Maritime-Gazetteer",
            "default_type": "vesselType"
        },
        # Min. length of mentions extracted by NER models.
        # This could be set at entity-type level, too - see example in "cancer".
        "min_mention_length": 2,
        # Set of ignored mentions that can be artifacts of the NER models.
        # This can be defined in general or at entity-type level (see example in cancer domain)
        "blocked_mentions": {},
        # Model configurations
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
        
        # === Entity Linking Configuration ===
        "linking_strategy": "reranker",
        "el_config": {
            "taxonomy_path": "taxonomies/maritime/VesselTypes.tsv",
            "taxonomy_source": "VesselTypes",
            "linker_type": "reranker",
            "el_model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.80,
            "context_window": 5,
            "max_contexts": 5,
            "use_sentence_context": False,
            "reranker_llm": "Qwen/Qwen3-1.7B",
            "reranker_top_k": 7,
            "reranker_fallbacks": True,
        },
        
        # === Type Matching Configuration ===
        "enforce_type_match": True,
        "taxonomy_type_column": "type",
        "type_mappings": {
            # Maps taxonomy type → expected NER type
            "vesselType": "vesseltype",
            "vessel": "vesseltype",
            "ship": "vesseltype",
            "boat": "vesseltype",
            # Specific vessel categories
            "cargo": "vesseltype",
            "tanker": "vesseltype",
            "passenger": "vesseltype",
            "fishing": "vesseltype",
            "military": "vesseltype",
        },
    },

    #### CANCER ####
    "cancer": {
        # Disable combined gazetteer - use FTS5 per entity type instead
        "gazetteer": {
            "enabled": False,
        },
        # Global minimum length for NER-extracted mentions (all entity types).
        # This could also be set per-entity type:
        # For example:
        #"min_mention_length": {
        #    "gene": 2,        # Gene symbols can be short (P53)
        #    "species": 3,     # Species names longer
        #    "disease": 3,     # Disease names longer
        #    "_default": 2,    # Fallback for unlisted types
        #},
        "min_mention_length": 2,
        # Model configurations        
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
        
        # === Entity Linking Configuration (FTS5) ===
        # FTS5-based linking strategy - uses exact matching, no threshold needed
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
            },
            "disease": {
                "index_path": "indices/cancer/doid_disease.db",
                "taxonomy_source": "DOID",
                "taxonomy_path": "taxonomies/cancer/DOID_DISEASE.tsv",
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
        
        # el_config for FTS5 - minimal config since exact matching is used
        # These are only used if semantic fallback is enabled for any entity type
        "el_config": {
            "linker_type": "fts5",  # Explicitly set
            # Fallback config (only used if fts5_linkers has "fallback": "semantic")
            "el_model_name": "intfloat/multilingual-e5-large-instruct",
            "threshold": 0.60,  # Lower threshold for semantic fallback
            "context_window": 3,
            "max_contexts": 3,
        },

        # === Type Matching Configuration ===
        # For cancer with FTS5, type matching is implicit (routing by entity type)
        # We set it to False since each FTS5 index only contains its type
        "enforce_type_match": False,
        "type_mappings": {
            # These mappings are provided for documentation/future use
            # They match the FTS5 linker routing
            "Gene": "gene",
            "Disease": "disease",
            "CellLine": "cellline",
            "Chemical": "chemical",
            "Species": "species",
            "Variant": "variant",
        },

        # Set of ignored mentions that can be artifacts of the NER models.
        # This could be defined as a set for all entities or at entity-level, as in this case.
        # Per-entity-type blocked mentions
        "blocked_mentions": {
                "species": {
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
                    # Other noise/false positives
                    "ad", "nod", "gas", "waterpipe", "spions",
                },
                "disease": {
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
    },
}
