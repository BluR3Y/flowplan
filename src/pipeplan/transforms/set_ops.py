from .base_cls import TransformOps
from typing import Callable, Dict
import pandas as pd

Ops = Callable[[pd.DataFrame, Dict], pd.DataFrame]
class SetTransform(TransformOps, type_id="set"):
    pass

@SetTransform.register_operation("filter_rows")
def filter_rows(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Example function matching the OpsFn signature."""
    col = params.get("column")
    threshold = params.get("threshold", 0)

    if col in df.columns:
        return df[df[col] > threshold]
    return df

if __name__ == "__main__":
    fn = SetTransform.get_operation("filter_rows")
    test_df = pd.DataFrame({"A": [1,5,10], "B": [2,2,2]})

    result = fn(test_df, {"column": "A", "threshold": 4})
    print("SetTransform Result:\n", result)