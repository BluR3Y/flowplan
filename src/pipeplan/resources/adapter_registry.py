from pipeplan.operation import Operation
from typing import Type

class ResourceOps(Operation, type_id="resource"):
    
    def __init_subclass__(cls, type_id, **kwargs):
        return super().__init_subclass__(type_id=f"resource.{type_id}", **kwargs)
    
    @classmethod
    def get_subclass(cls, type_id: str) -> Type["ResourceOps"]:
        return cls.get_operation_type(f"resource.{type_id}")

# --- Intermediate Resource Namespaces ---
class FileResource(ResourceOps, type_id="file"): pass
class DatabaseResource(ResourceOps, type_id="db"): pass
class APIResource(ResourceOps, type_id="api"): pass