from .base_adapter import PipelineResource
from typing import Dict, List, Type
import pandas as pd
import json

class JsonAdapter(PipelineResource, type_id="json"):
    
    def extract_data(self, records: str = None) -> pd.DataFrame:
        with open(self.cfg.get("path"), 'r') as file:
            data = json.load(file)
            if isinstance(data, list):
                return pd.DataFrame(data)
            return pd.DataFrame(data.get(records) if records else data)
    
    def load_data(self, df: pd.DataFrame, records: str = None):
        with open(self.cfg.get("path"), 'w') as file:
            json.dump(df.to_dict(), file, indent=4)