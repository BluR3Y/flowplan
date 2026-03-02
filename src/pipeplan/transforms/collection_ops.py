from .base_cls import TransformOps
from typing import Callable, Dict, List
import pandas as pd

OpsFn = Callable[[List[pd.DataFrame], Dict], pd.DataFrame]
class CollectionTransform(TransformOps, type_id="collection"):
    pass

@CollectionTransform.register_operation("test_fn")
def test_fn(dfs: List[pd.DataFrame], params: Dict) -> pd.DataFrame:
    test_data = {
        "A": [1, 4, 10],
        "B": [2, 2, 2]
    }
    return pd.DataFrame(test_data)

if __name__ == "__main__":
    fn = CollectionTransform.get_operation("test_fn")
    result = fn([], {})
    print("CollectionTransform Result:\n", result)