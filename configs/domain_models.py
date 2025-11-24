"""
Domain-specific model configuration for SciLake NER/NEL enrichment.
Each domain defines:
 - models: list of NER models to apply (GLiNER / RoBERTa)
 - labels: label sets per model type
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
                #"threshold": 0.99,
                "threshold": 0.95,
            },
        ],
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
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-CCAM-roberta-large-all",
                "type": "roberta",
                "threshold": 0.7,
            }
        ],
        "labels": {
            "roberta": [
                "vehicleType",
                "VRUType",
                "scenarioType",
                "communicationType",
                "entityConnectionType",
                "levelOfAutomation",
                "sensorType",
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
                "threshold": 0.85
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
            },
        ],
        "labels": {
            "roberta": ["vesselType"],
        },
        "kb": {
            "default": "Maritime ontology provided by partners",
        },
    },

    # !!!!!!!!!! UPDATE !!!!!!!!!!!!!
    "cancer": {
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Biomed-roberta-large",
                "type": "roberta",
                "threshold": 0.8,
            },
        ],
        "labels": {
            "roberta": [
                "Gene",
                "Disease",
                "CellLine",
                "Chemical",
                "Species",
                # "Variant" (ignored for now)
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
