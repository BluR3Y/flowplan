from ..base_level import TransformTask
from typing import Callable, Dict, Any, List
from importlib.metadata import entry_points
import pandas as pd
import logging

log = logging.getLogger(__name__)

TransformFn = Callable[[[pd.DataFrame], Dict], pd.DataFrame]
class CollectionTransform(TransformTask, type_id="collection"):
    _COLLECTION_OPERATION_REGISTRY: Dict[str, TransformFn] = {}

    @classmethod
    def register_operation(cls, name: str):
        """Decorator to register a transform operation."""
        def decorator(fn: TransformFn):
            if name in cls._COLLECTION_OPERATION_REGISTRY:
                raise ValueError(f"Collection transform operation '{name}' already exists.")
            cls._COLLECTION_OPERATION_REGISTRY[name] = fn
        return decorator

    @classmethod
    def get_operation(cls, name: str):
        """Retrieve a transform operation by name"""
        if name not in cls._COLLECTION_OPERATION_REGISTRY:
            cls._load_plugins("collection_transforms", cls._COLLECTION_OPERATION_REGISTRY)
        if name not in cls._COLLECTION_OPERATION_REGISTRY:
            raise KeyError(f"Unknown collection transform: {name}")
        
        return cls._COLLECTION_OPERATION_REGISTRY[name]