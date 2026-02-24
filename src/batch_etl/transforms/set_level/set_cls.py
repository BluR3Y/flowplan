from ..base_level import TransformTask
from typing import Callable, Dict, Any, List
from importlib.metadata import entry_points
import pandas as pd
import logging

log = logging.getLogger(__name__)

TransformFn = Callable[[pd.DataFrame, Dict], pd.DataFrame]
class SetTransform(TransformTask, type_id="set"):
    _SET_OPERATION_REGISTRY: Dict[str, TransformFn] = {}

    @classmethod
    def register_operation(cls, name: str):
        """Decorator to register a transform operation."""
        def decorator(fn: TransformFn):
            if name in cls._SET_OPERATION_REGISTRY:
                raise ValueError(f"Set transform operation '{name}' already exists.")
            cls._SET_OPERATION_REGISTRY[name] = fn
        return decorator
    
    @classmethod
    def get_operation(cls, name: str):
        """Retrieve a transform operation by name"""
        if name not in cls._SET_OPERATION_REGISTRY:
            # cls._load_plugins()
            cls._load_plugins("set_transforms", cls._SET_OPERATION_REGISTRY)
        if name not in cls._SET_OPERATION_REGISTRY:
            raise KeyError(f"Unknown set transform: {name}")
        
        return cls._SET_OPERATION_REGISTRY[name]