from typing import Callable, Dict, Any
import pandas as pd
from importlib.metadata import entry_points

from ..exceptions import TransformError

TransformFn = Callable[[pd.Series, dict], pd.Series]
_REGISTRY: Dict[str, TransformFn] = {}

def register_transform(name: str):
    def decorator(fn: TransformFn):
        _REGISTRY[name] = fn
    return decorator

def get_transform(name: str) -> TransformFn:
    if name not in _REGISTRY:
        # Lazy load plugins
        _load_plugins()
    if name not in _REGISTRY:
        raise TransformError(f"Unknown transform: {name}")
    return _REGISTRY[name]

def _load_plugins():
    try:
        eps = entry_points()
        # Support python 3.8+ select vs older dict interface
        candidates = eps.select(group="batch_etl.transforms") if hasattr(eps, "select") else eps.get("batch_etl.transforms", [])
        for ep in candidates:
            _REGISTRY[ep.name] = ep.load()
    except Exception:
        pass

# --- Apply Pipeline ---

def apply_pipeline(series: pd.Series, steps: list[dict] | dict) -> pd.Series:
    if not steps:
        return series
    if isinstance(steps, dict):
        steps = [steps]
    
    # Flatten logic for nested lists
    def _flatten(seq):
        for x in seq:
            if isinstance(x, list): yield from _flatten(x)
            else: yield x
    
    steps = list(_flatten(steps))
    out = series
    for step in steps:
        if not isinstance(step, dict) or len(step) != 1:
            raise TransformError(f"Invalid transform step: {step}")
        name, params = next(iter(step.items()))
        fn = get_transform(name)
        out = fn(out, params or {})
    return out