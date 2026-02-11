import pandas as pd
from .registry import register_transform
from ..exceptions import TransformError

@register_transform("cast")
def _cast(series: pd.Series, params: dict) -> pd.Series:
    """
    Cast series to a specific data type.
    Params:
        to (str): Target type (integer, number, string, boolean, date).
        format (str): Format string for date casting.
        on_cast_error (str): 'fail', 'coerce_null', 'drop_row'.
    """
    target = params.get("to")
    policy = params.get("on_cast_error", "fail")

    try:
        pass
    except Exception:
        if policy == "coerce_null":
            return pd.Series([None] * len(series), index=series.index, dtype="object")
        if policy == "drop_row":
            # last here