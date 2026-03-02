from pipeplan.operation import Operation
from typing import Type, Dict, Any
from pipeplan.resources import ResourceOps, Resource
import pandas as pd

class TransferOps(Operation, type_id="transfer"):
    """Namespace for generic transfer actions (extract, load)."""

    _CONNECTION_REGISTRY: Dict[str, Resource] = {}

    @classmethod
    def register_resource(cls, id: str, resource: Resource):
        if id:
            if id in cls._CONNECTION_REGISTRY:
                raise ValueError(f"Connection '{id}' already established.")
            cls._CONNECTION_REGISTRY[id] = resource
    
    @classmethod
    def get_resource(cls, id: str) -> Resource:
        if id not in cls._CONNECTION_REGISTRY:
            raise KeyError(f"Unknown resource connection: {id}")
        return cls._CONNECTION_REGISTRY[id]
    
    @classmethod
    def clear_connections(cls):
        """
        Safely closes and removes all connections.
        Crucial for avoiding global state leaks across different pipeline runs.
        """
        for res_id, conn in cls._CONNECTION_REGISTRY.items():
            conn.disconnect()
        cls._CONNECTION_REGISTRY.clear()

# --- Registered Operations ---

@TransferOps.register_operation("extract")
def _extract(resource: str, *args, **kwargs) -> Dict[str, Any]:
    adapter = TransferOps.get_resource(resource)
    with adapter:
        data = adapter.read(*args, **kwargs)
        data = pd.json_normalize(data)
    return pd.DataFrame(data)

@TransferOps.register_operation("load")
def _load(resource: str, data: pd.DataFrame, *args, **kwargs) -> None:
    adapter = TransferOps.get_resource(resource)
    with adapter:
        adapter.write(data=data.to_dict(), *args, **kwargs)

if __name__ == "__main__":
    res_sub = ResourceOps.get_subclass("file")
    res_type = res_sub.get_operation("json")
    res_instance = res_type(path="C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json")
    TransferOps.register_resource("test_conn_1", res_instance)
    extract_fn = TransferOps.get_operation("extract")
    extract_outcome: pd.DataFrame = extract_fn("test_conn_1")
    print(extract_outcome.columns)