import pandas as pd
import re
from .registry import register_transform
from ..utils.text_normalize import normalize_series
from ..exceptions import TransformError

@register_transform("regex_replace")
def _regex_replace(series: pd.Series, params: dict) -> pd.Series:
    pat = params.get("pattern")
    repl = params.get("repl", "")
    flags = params.get("flags", "")
    if pat is None:
        raise TransformError("regex_replace requires `pattern`.")
    re_flags = 0
    if "i" in flags: re_flags |= re.IGNORECASE
    s= series.astype("string")
    mask = s.notna()
    s.loc[mask] = s.loc[mask].str.replace(pat, repl, regex=True, flags=re_flags)
    return s

@register_transform("cast")
def _cast(series: pd.Series, params: dict) -> pd.Series:
    target = params.get("to")
    policy = params.get("on_cast_error", "fail")    # "fail" | "coerce_null" | "drop_row"
    try:
        if target == "integer":
            out = pd.to_numeric(series, errors="raise").astype("Int64")
        elif target == "number":
            out = pd.to_numeric(series, errors="raise").astype(float)
        elif target == "string":
            out = series.astype("string")
        elif target == "boolean":
            out = series.astype("boolean")
        elif target == "date":
            fmt = params.get("format")  # e.g., "%Y-%m-%d"
            out = pd.to_datetime(series, format=fmt, errors="raise")
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

@register_transform("normalize")
def _normalize(series: pd.Series, steps: list) -> pd.Series:
    return normalize_series(series, steps)

@register_transform("map")
def _map(series: pd.Series, params: dict) -> pd.Series:
    if not isinstance(params, dict):
        raise TransformError("map transform requires a dict")
    return series.map(params).fillna(series)

@register_transform("affix")
def _affix(series: pd.Series, params: dict) -> pd.Series:
    text = str(params.get("text", ""))
    position = params.get("position", "suffix") # "prefix" or "suffix"
    s = series.astype("string").fillna("")
    if position == "prefix": return text + s
    return s + text

@register_transform("strftime")
def _strftime(series: pd.Series, params: dict) -> pd.Series:
    fmt = params.get("format", "%Y-%m-%d")
    return pd.to_datetime(series, errors="coerce").dt.strftime(fmt)