from abc import ABC, abstractmethod
from typing import Dict, Type, Any
import pandas as pd

class PipelineResource(ABC):
    _RESOURCE_REGISTRY: Dict[str, Type["PipelineResource"]] = {}

    def __init__(self, resource_cfg: Dict[str, Any]):
        self.cfg = resource_cfg

    def __init_subclass__(cls, type_id: str):
        if type_id:
            if type_id in cls._RESOURCE_REGISTRY:
                raise ValueError(f"Resource adapter '{type_id}' already registered.")
            cls._RESOURCE_REGISTRY[type_id] = cls
    
    @classmethod
    def get_resource_type(cls, type_id: str) -> Type["PipelineResource"]:
        if type_id not in cls._RESOURCE_REGISTRY:
            raise ValueError(f"Unknown resource adapter: {type_id}")
        return cls._RESOURCE_REGISTRY[type_id]
    
    @abstractmethod
    def extract_data(self, records: str = None) -> pd.DataFrame:
        ...
    
    @abstractmethod
    def load_data(self, df: pd.DataFrame, records: str = None):
        ...