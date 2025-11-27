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
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Biomed-roberta-large",
                "type": "roberta",
                "threshold": 0.8,
                # RoBERTa labels - DOCUMENTATION ONLY
                "output_labels": [
                    "Gene", "Disease", "CellLine", 
                    "Chemical", "Species",
                ],
            },
        ],
        "labels": {
            "roberta": [
                "Gene", "Disease", "CellLine",
                "Chemical", "Species",
            ],
        },
        "kb": {
            "Gene": "NCBI Gene",
            "Disease": "Disease Ontology (fallback: MeSH)",
            "CellLine": "BRENDA",
            "Chemical": "DrugBank",
            "Species": "NCBI Taxonomy",
        },
    },
}
