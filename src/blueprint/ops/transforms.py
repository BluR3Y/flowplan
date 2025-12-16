import pandas as pd
import re
import unicodedata
from typing import Callable, Iterable, Dict, Any
from blueprint.exceptions import TransformError
from blueprint.utils import load_entrypoints

Transform = Callable[[pd.Series, dict], pd.Series]
TRANSFORM_PLUGINS = load_entrypoints("blueprint.transforms")

def _regex_replace(series: pd.Series, params: dict) -> pd.Series:
    pat = params.get("pattern")
    repl = params.get("repl", "")
    flags = params.get("flags", "")
    if pat is None:
        raise TransformError("regex_replace requires `pattern`.")
    re_flags = 0
    if "i" in flags: re_flags |= re.IGNORECASE
    s = series.astype("string")
    mask = s.notna()
    s.loc[mask] = s.loc[mask].str.replace(pat, repl, regex=True, flags=re_flags)
    return s

def _cast(series: pd.Series, params: dict) -> pd.Series:
    target = params.get("to")
    policy = params.get("on_cast_error", "fail")
    try:
        if target == "integer": out = pd.to_numeric(series, errors="raise").astype("Int64")
        elif target == "number": out = pd.to_numeric(series, errors="raise").astype(float)
        elif target == "string": out = series.astype("string")
        elif target == "boolean": out = series.astype("boolean")
        elif target == "date":
            fmt = params.get("format")
            out = pd.to_datetime(series, format=fmt, errors="raise" if policy == "fail" else "coerce")
        else:
            raise TransformError(f"Unknown cast target: {target}")
        return out
    except Exception:
        if policy == "coerce_null":
            return pd.Series([pd.NA if v is not None else None for v in series], index=series.index)
        if policy == "drop_row":
            series.attrs["__drop__"] = True
            return series
        raise

def _normalize(series: pd.Series, steps: list[str]) -> pd.Series:
    series = series.astype(str)
    for step in steps:
        if step == "strip": series = series.str.strip()
        elif step == "lower": series = series.str.lower()
        elif step == "upper": series = series.str.upper()
        elif step == "title": series = series.str.title()
        else: raise ValueError(f"Unsupported operation: {step}")
    return series

def _map(series: pd.Series, params: dict) -> pd.Series:
    return series.map(params.get("map", {})).fillna(series)

def _affix(series: pd.Series, params: dict) -> pd.Series:
    text = str(params.get("text", ""))
    position = params.get("position", "suffix")
    series = series.astype("string").fillna("")
    return (text + series) if position == "prefix" else (series + text)

def _strftime(series: pd.Series, params: dict) -> pd.Series:
    fmt = params.get("format", "%Y-%m-%d")
    return pd.to_datetime(series, errors="coerce").dt.strftime(fmt)

REGISTRY: Dict[str, Transform] = {
    "regex_replace": _regex_replace,
    "cast": _cast,
    "normalize": _normalize,
    "map": _map,
    "affix": _affix,
    "strftime": _strftime
}
REGISTRY.update(TRANSFORM_PLUGINS)

def apply_pipeline(series: pd.Series, steps: list[dict]) -> pd.Series:
    if not steps: return series
    if isinstance(steps, dict): steps = [steps]
    
    # Flatten Refs
    def _flatten(seq):
        for x in seq:
            yield from (_flatten(x) if isinstance(x, list) else [x])
    
    flat_steps = list(_flatten(steps))
    out = series
    for step in flat_steps:
        name, params = next(iter(step.items()))
        fn = REGISTRY.get(name)
        if not fn: raise TransformError(f"Unknown transform `{name}`")
        out = fn(out, params or {})
    return out