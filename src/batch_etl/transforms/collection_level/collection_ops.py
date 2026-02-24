from .collection_cls import CollectionTransform
import pandas as pd
from typing import Dict
from ...exceptions import TransformError

@CollectionTransform.register_operation("merge")
def _merge(dfs: Dict[str, pd.DataFrame], *args, **kwargs) -> pd.DataFrame:
    # Fill Login at later session
    return dfs[0]