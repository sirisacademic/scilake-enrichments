"""
Domain-specific model configuration for SciLake NER/NEL enrichment.
Each domain defines:
 - models: list of NER models to apply (GLiNER / RoBERTa)
 - labels: label sets per model type
 - kb: knowledge base(s) to use for Entity Linking
"""

DOMAIN_MODELS = {
    "neuro": {
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Neuroscience-GLiNER-large",
                "type": "gliner",
                "threshold": 0.99,
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
            },
            {
                "name": "SIRIS-Lab/SciLake-CCAM-GLiNER-large-all",
                "type": "gliner",
                "threshold": 0.6,
                "for": "scenarioType",  # secondary model focused on scenarioType
            },
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
            "gliner": ["ScenarioType"],
        },
        "kb": {
            "default": "Project-specific CCAM vocabulary (pilot sheet)",
        },
    },

    "energy": {
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Energy-GLiNER-medium",
                "type": "gliner",
                "threshold": 0.9,
                "for": "energyStorage",
            },
            {
                "name": "SIRIS-Lab/SciLake-Energy-roberta-base",
                "type": "roberta",
                "threshold": 0.85,
                "for": "energyType",
            },
        ],
        "labels": {
            "gliner": ["EnergyStorage"],
            "roberta": ["EnergyType"],
        },
        "kb": {
            "default": "IRENA energy taxonomy",
        },
    },

    "maritime": {
        "models": [
            {
                "name": "SIRIS-Lab/SciLake-Maritime-roberta-base",
                "type": "roberta",
                "threshold": 0.75,
            },
        ],
        "labels": {
            "roberta": ["VesselType"],
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