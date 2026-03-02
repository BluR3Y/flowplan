from .adapter_registry import ResourceOps, FileResource, DatabaseResource, APIResource
from .base_adapter import Resource
from .file_adapters import json_adapter,  excel_adapter
# from .file_adapter import FileResource
# from .db_adapter import DatabaseResource

__all__ = ["ResourceOps", "Resource", "FileResource", "DatabaseResource", "APIResource"]

if __name__ == "__main__":
    resource_type = ResourceOps.get_operation_type("resource")
    resource_sub = resource_type.get_subclass("file")
    resource_adapter = resource_sub.get_operation("json")
    adapter_instance = resource_adapter(path="C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json")
    adapter_instance.__enter__()
    print(adapter_instance.extract())
    adapter_instance.__exit__(None,None,None)