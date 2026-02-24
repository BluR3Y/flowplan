from .set_cls import SetTransform
import pandas as pd
from ...exceptions import TransformError

@SetTransform.register_operation("filter")
def _filter(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
    # Fill in logic in later session
    return df