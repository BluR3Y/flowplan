from typing import Callable, Dict, Any, List
import pandas as pd
from importlib.metadata import entry_points
import logging

from ..exceptions import TransformError

log = logging.getLogger(__name__)

TransformFn = Callable[[pd.Series, dict], pd.Series]
_REGISTRY: Dict[str, TransformFn] = {}

def register_transform(name: str):
    """Decorator to register a transform function."""
    def decorator(fn: TransformFn):
        if name in _REGISTRY:
            log.warning(f"Overwriting existing transform '{name}' with {fn.__name__}")
        _REGISTRY[name] = fn
    return decorator

def _load_plugins():
    """Load external plugins registered via entry points."""
    try:
        eps = entry_points()
        group_name = "batch_etl.transforms"

        # Support Python 3.10+ select() and older dict interface
        candidates = eps.select(group=group_name) if hasattr(eps, "select") else eps.get(group_name, [])

        for ep in candidates:
            if ep.name not in _REGISTRY:
                try:
                    # Load plugin (executes module level code/decorators)
                    plugin_fn = ep.load()

                    # If the plugin didn't use the decorator but just returned a callable.
                    # we register it manually here
                    if ep.name not in _REGISTRY and callable(plugin_fn):
                        _REGISTRY[ep.name] = plugin_fn
                    
                    log.debug(f"Loaded transform plugin: {ep.name}")
                except Exception as e:
                    log.warning(f"Failed to load plugin {ep.name}: {e}")
    except Exception as e:
        log.debug(f"Plugin loading skipped/failed: {e}")

def get_transform(name: str) -> TransformFn:
    """Retrieve a transform function by name, loading built-ins/plugins if needed."""
    if name not in _REGISTRY:
        _load_plugins()
    
    if name not in _REGISTRY:
        raise TransformError(f"Unknown transform: {name}")

    return _REGISTRY[name]

def apply_transformations(df: pd.DataFrame, steps: list[dict]) -> pd.DataFrame:
    out = df.copy()
    for step in steps:
        op = step.get("op")
        params = step.get("params")
        fn = get_transform(op)
        out
    return out