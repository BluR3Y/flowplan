from .element_cls import ElementOperation
import pandas as pd
from typing import Union
from ...exceptions import TransformError

ElementOperation.register_operation("cast")
def _cast(series: pd.Series, *args, **kwargs):
    target = kwargs.get("to") if not args else args[0]
    policy = kwargs.get("on_error", "fail") # "fail" | "coerce"

    try:
        if target in ["string", "boolean"]:
            return series.astype(target)
        
        on_error = "coerce" if policy == "coerce" else "raise"
        if target == "integer":
            out = pd.to_numeric(series, errors=on_error).astype("Int64")
        elif target == "float":
            out = pd.to_numeric(series, errors=on_error).astype("float")
        elif target == "date":
            fmt = kwargs.get("format")
            out = pd.to_datetime(series, format=fmt, errors=on_error)
        else:
            raise TransformError(f"Unknown cast target: {target}")
        return out
    except Exception:
        if policy == "coerce":
            return pd.Series([pd.NA if v is not None else None for v in series], index=series.index)
        raise