from .base_level import TransformOperation
from typing import Callable, Dict, Any, List
from importlib.metadata import entry_points
import pandas as pd
import logging

log = logging.getLogger(__name__)

TransformFn = Callable[[[pd.DataFrame], Dict], pd.DataFrame]
class CollectionOperation(TransformOperation, type_id="collection"):
    _COLLECTION_OPERATION_REGISTRY: Dict[str, TransformFn] = {}

    def register_operation(self, name: str):
        """Decorator to register a transform operation."""
        def decorator(fn: TransformFn):
            if name in self._COLLECTION_OPERATION_REGISTRY:
                raise ValueError(f"Collection transform operation '{name}' already exists.")
            self._COLLECTION_OPERATION_REGISTRY[name] = fn
        return decorator

    def _load_plugins(self):
        """Load external plugins registered via entry points."""
        try:
            eps = entry_points()
            group_name = "batch_etl.collection_transforms"

            # Support Python 3.10+ select() and older dict interface
            candidates = eps.select(group=group_name) if hasattr(eps, "select") else eps.get(group_name, [])

            for ep in candidates:
                if ep.name not in self._COLLECTION_OPERATION_REGISTRY:
                    try:
                        # Load plugin (executes module level code/decorators)
                        plugin_fn = ep.load()

                        # If the plugin didn't use the decorator but just returned a callable.
                        # we register it manually here
                        if ep.name not in self._COLLECTION_OPERATION_REGISTRY and callable(plugin_fn):
                            self._COLLECTION_OPERATION_REGISTRY[ep.name] = plugin_fn
                        
                        log.debug(f"Loaded collection transform plugin: {ep.name}")
                    except Exception as e:
                        log.warning(f"Failed to load collection plugin {ep.name}: {e}")
        except Exception as e:
            log.debug(f"Plugin loading skipped/failed: {e}")

    def get_operation(self, name):
        """Retrieve a transform operation by name"""
        if name not in self._COLLECTION_OPERATION_REGISTRY:
            self._load_plugins()
        if name not in self._COLLECTION_OPERATION_REGISTRY:
            raise KeyError(f"Unknown collection transform: {name}")
        
        return self._COLLECTION_OPERATION_REGISTRY[name]