from .base_adapter import PipelineResource
from abc import ABC
from typing import Dict, Type

class ResourceManager(ABC):
    _CONNECTIONS: Dict[str, Type["PipelineResource"]] = None

    def __init__(self):
        self._CONNECTIONS = {}
    
    def create_connection(self, connection_id: str, resource: Type["PipelineResource"]):
        if connection_id:
            if connection_id in self._CONNECTIONS:
                raise ValueError(f"Resource connection '{connection_id}' already established.")
            self._CONNECTIONS[connection_id] = resource
    
    def get_connection(self, connection_id: str):
        if connection_id not in self._CONNECTIONS:
            raise ValueError(f"Unknown resource connection: {connection_id}")
        return self._CONNECTIONS[connection_id]
    

if __name__ == "__main__":
    test_conn = ResourceManager.get_connection