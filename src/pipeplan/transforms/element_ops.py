from .base_cls import TransformOps
from typing import Callable, Dict
import pandas as pd

OpsFn = Callable[[pd.Series, Dict], pd.Series]

class ElementTransform(TransformOps, type_id="element"):
    pass

@ElementTransform.register_operation("clean_strings")
def clean_strings(series: pd.Series, params: Dict) -> pd.Series:
    """Example function matching the OpsFn signature."""
    to_replace = params.get("replace", "")
    return series.astype(str).str.replace(to_replace, "", regex=False).str.strip()

if __name__ == "__main__":
    fn = ElementTransform.get_operation("clean_strings")
    test_series = pd.Series(["  apple  ", "banana_bad", "cherry"])

    result = fn(test_series, {"replace": "_bad"})
    print("ElementOps Result:")
    print(result.tolist())