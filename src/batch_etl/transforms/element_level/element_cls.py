from ..base_level import TransformTask
from typing import Callable, Dict, Any, List
from importlib.metadata import entry_points
import pandas as pd
import logging

log = logging.getLogger(__name__)

TransformFn = Callable[[pd.Series, Dict], pd.Series]
class ElementTransform(TransformTask, type_id="element"):
    _ELEMENT_OPERATION_REGISTRY: Dict[str, TransformFn] = {}

    @classmethod
    def register_operation(cls, name: str):
        """Decorator to register a transform operation."""
        def decorator(fn: TransformFn):
            if name in cls._ELEMENT_OPERATION_REGISTRY:
                raise ValueError(f"Element transform operation '{name}' already exists.")
            cls._ELEMENT_OPERATION_REGISTRY[name] = fn
        return decorator
    
    @classmethod
    def get_operation(cls, name: str):
        """Retrieve a transform operation by name"""
        if name not in cls._ELEMENT_OPERATION_REGISTRY:
            cls._load_plugins("element_transforms", cls._ELEMENT_OPERATION_REGISTRY)
        if name not in cls._ELEMENT_OPERATION_REGISTRY:
            raise KeyError(f"Unknown element transform: {name}")
        
        return cls._ELEMENT_OPERATION_REGISTRY[name]
    
    @classmethod
    def apply_operation(cls, **kwargs) -> pd.Series:
        # Last Here
        pass