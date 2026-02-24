from ..etl_task import ETLTask
from ..utils import filter_missing_keys
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable
import pandas as pd

ExtractFn = Callable[[Dict], pd.DataFrame]
class ExtractTask(ETLTask):
    _ADAPTER_REGISTRY: Dict[str, ExtractFn] = {}

    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def register_adapter(cls, name: str):
        """Decorator to register an extract adapter."""
        def decorator(fn: ExtractFn):
            if name in cls._ADAPTER_REGISTRY:
                raise ValueError(f"Extract adapter '{name}' already exists.")
            cls._ADAPTER_REGISTRY[name] = fn
        return decorator
    
    @classmethod
    def get_adapter(cls, name: str):
        """Retrieve an extract adapter by name."""
        if name not in cls._ADAPTER_REGISTRY:
            cls._load_plugins("extract_adapter", cls._ADAPTER_REGISTRY)
        if name not in cls._ADAPTER_REGISTRY:
            raise ValueError(f"Unknown extract adapter: {name}")
        return cls._ADAPTER_REGISTRY[name]
    
    def _process_data(self):
        pass

    def run_task(self):
        connect_config = self.cfg.get("connect", {})
        missing_params = filter_missing_keys(connect_config, ["adapter", "params"])
        if missing_params:
            raise ValueError(f"Extract task '{self.id}' is missing connection configuration: {', '.join(missing_params)}")
        extract_adapter = self.get_adapter(connect_config.get("adapter"))
        extract_params = connect_config.get("params")
        print(extract_adapter(**extract_params))