from abc import ABC, abstractmethod
from .resources import PipelineResource
from typing import Dict, Type

class ResourceManager(ABC):
    _CONNECTION_REGISTRY: Dict[str, Type["PipelineResource"]] = None

    def __init__(self):
        self._CONNECTION_REGISTRY = {}
    
    def register_connection(self, connection_id: str, resource: PipelineResource):
        if connection_id:
            if connection_id in self._CONNECTION_REGISTRY:
                raise ValueError(f"Resource connection '{connection_id}' already established.")
            self._CONNECTION_REGISTRY[connection_id] = resource

    def get_connection(self, connection_id: str) -> Type["PipelineResource"]:
        if connection_id not in self._CONNECTION_REGISTRY:
            raise ValueError(f"Unknown resource connection: {connection_id}")
        return self._CONNECTION_REGISTRY[connection_id]
        
if __name__ == "__main__":
    my_connections = {
        "connections": [
            {
                "connection_id": "test_log_conn",
                "resource": "json",
                "params": {
                    "path": "D:/data/json_data.json"
                }
            },
            {
                "connection_id": "test_db_conn",
                "resource": "sql",
                "params": {
                    "adapter": "access",
                    "path": "D:/data/access_data.accdb"
                }
            }
        ],
        "tasks": [
            {
                "task_id": "extract_json_1",
                "phase": "extract",
                "configure": {
                    "records": "Activty_Types"
                }
            }
        ]
    }
    conn_manager = ResourceManager()
    for conn in my_connections.get("connections", []):
        conn_cls = PipelineResource.get_resource_type(conn.get("resource"))
        conn_obj = conn_cls(conn.get("params", {}))
        conn_manager.register_connection(conn.get("connection_id"), conn_obj)
    
    conn_hook = conn_manager.get_connection("test_log_conn")
    data = conn_hook.extract_data()