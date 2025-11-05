import os
import pandas as pd
from flashtext import KeywordProcessor
from typing import List, Dict, Any
from pathlib import Path

class GazetteerLinker:
    def __init__(self, taxonomy_path: str):
        # Resolve relative to project root
        if not os.path.isabs(taxonomy_path):
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        self.taxonomy_df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        self._build_index()
        
    
    def _build_index(self):
        """Build FlashText index with all terms and aliases."""
        for _, row in self.taxonomy_df.iterrows():
            metadata = {
                'irena_id': str(row['id']),
                'concept': row['concept'],
                'wikidata_id': row.get('wikidata_id', ''),
                'type': row.get('type', '')
            }
            
            # Add primary concept
            self.keyword_processor.add_keyword(row['concept'], metadata)
            
            # Add aliases
            if pd.notna(row.get('wikidata_aliases')):
                for alias in row['wikidata_aliases'].split(' | '):
                    alias = alias.strip()
                    if alias:
                        self.keyword_processor.add_keyword(alias, metadata)
    
    def extract_entities(
        self, 
        text: str, 
        section_id: str, 
        domain: str = "energy"
    ) -> List[Dict[str, Any]]:
        """Extract gazetteer matches from text."""
        matches = self.keyword_processor.extract_keywords(text, span_info=True)
        
        entities = []
        for metadata, start, end in matches:
            entity = {
                "entity": self._map_type(metadata['type']),
                "text": text[start:end],
                "score": 1.0,  # Exact match
                "start": start,
                "end": end,
                "model": "IRENA-Gazetteer",
                "domain": domain,
                "section_id": section_id,
                "linking": self._create_linking(metadata)
            }
            entities.append(entity)
        
        return entities
    
    def _map_type(self, irena_type: str) -> str:
        """Map IRENA type to NER entity type."""
        type_map = {
            'Renewables': 'energytype',
            'Non-renewable': 'energytype',
            'Storage': 'energystorage'
        }
        return type_map.get(irena_type, 'energytype')
    
    def _create_linking(self, metadata: Dict) -> List[Dict[str, str]]:
        """Create linking structure."""
        linking = [{
            "source": "IRENA",
            "id": metadata['irena_id'],
            "name": metadata['concept']
        }]
        
        if metadata['wikidata_id']:
            wd_id = metadata['wikidata_id'].split('/')[-1]
            linking.append({
                "source": "Wikidata",
                "id": wd_id,
                "name": metadata['concept']
            })
        
        return linking
        

