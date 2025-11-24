import os
import pandas as pd
from flashtext import KeywordProcessor
from typing import List, Dict, Any
from pathlib import Path

class GazetteerLinker:
    def __init__(self, taxonomy_path: str, taxonomy_source: str = None, 
                 model_name: str = None, default_type: str = None, domain: str = None):
        # Resolve relative to project root
        if not os.path.isabs(taxonomy_path):
            project_root = Path(__file__).parent.parent
            taxonomy_path = project_root / taxonomy_path
        
        self.taxonomy_df = pd.read_csv(taxonomy_path, sep='\t').fillna('')
        self.keyword_processor = KeywordProcessor(case_sensitive=False)
        
        # Auto-detect or use provided values
        self.taxonomy_source = taxonomy_source or self._detect_source(taxonomy_path)
        self.model_name = model_name or f"{self.taxonomy_source}-Gazetteer"
        self.default_type = default_type
        self.domain = domain
        
        self._build_index()
        
    
    def _build_index(self):
        """Build FlashText index with all terms and aliases."""
        for _, row in self.taxonomy_df.iterrows():
            metadata = {
                'taxonomy_id': str(row['id']),
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
    
    def _detect_source(self, taxonomy_path: Path) -> str:
        """Auto-detect taxonomy source from filename."""
        filename = Path(taxonomy_path).stem
        
        # Map common patterns
        if 'IRENA' in filename:
            return 'IRENA'
        elif 'Neuroscience' in filename or 'UBERON' in filename:
            return 'OPENMINDS-UBERON'
        elif 'Vessel' in filename or 'Maritime' in filename:
            return 'Maritime-Ontology'
        else:
            # Use filename as fallback
            return filename
    
    def extract_entities(
        self, 
        text: str, 
        section_id: str, 
        domain: str = None
    ) -> List[Dict[str, Any]]:
        """Extract gazetteer matches from text."""
        domain = domain or self.domain
        matches = self.keyword_processor.extract_keywords(text, span_info=True)
        
        entities = []
        for metadata, start, end in matches:
            entity = {
                "entity": self._map_type(metadata['type'], domain),
                "text": text[start:end],
                "score": 1.0,  # Exact match
                "start": start,
                "end": end,
                "model": self.model_name,
                "domain": domain,
                "section_id": section_id,
                "linking": self._create_linking(metadata)
            }
            entities.append(entity)
        
        return entities
    
    def _map_type(self, taxonomy_type: str, domain: str = None) -> str:
        """Map taxonomy type to NER entity type based on domain."""
        # Use self.domain if domain not provided
        domain = domain or self.domain
        
        # Handle empty types - use default_type if available
        if not taxonomy_type or taxonomy_type.strip() == "":
            if self.default_type:
                return self.default_type
            # Fallback defaults by domain
            defaults = {
                'neuro': 'UBERONParcellation',
                'maritime': 'vesselType',
                'energy': 'energytype'
            }
            return defaults.get(domain, 'Unknown')
        
        # Domain-specific mappings for non-empty types
        if domain == 'energy':
            energy_map = {
                'Renewables': 'energytype',
                'Non-renewable': 'energytype',
                'Storage': 'energystorage'
            }
            return energy_map.get(taxonomy_type, 'energytype')
        
        elif domain == 'maritime':
            # All maritime types map to vesselType
            return 'vesselType'
        
        else:
            # For neuro and others, use type as-is
            return taxonomy_type
    
    def _create_linking(self, metadata: Dict) -> List[Dict[str, str]]:
        """Create linking structure."""
        linking = [{
            "source": self.taxonomy_source,
            "id": metadata['taxonomy_id'],
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
        

