from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd
from ..transforms.registry import apply_pipeline, get_transform
from ..utils.type_enforce import enforce_types, validate_frame

class SourceAdapter(ABC):
    def __init__(self, source_cfg: Dict[str, Any], aliases: Dict[str, Any]):
        self.cfg = source_cfg
        self.aliases = aliases
    
    @abstractmethod
    def load_data(self) -> dict[str, pd.DataFrame]:
        ...
    
    def _process_data(self, df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
        # 1. Column mapping & transform
        cols_map = data_cfg.get("columns", {})
        if cols_map:
            out = {}
            for src_col, spec in cols_map.items():
                alias = spec.get("alias") if isinstance(spec, dict) else None
                transforms = spec.get("transforms", []) if isinstance(spec, dict) else []
                col = df[src_col] if src_col in df.columns else pd.Series([None]*len(df))
                if transforms:
                    col = apply_pipeline(col, transforms)
                    # print(get_transform("regex_replace"))
                out[alias or src_col] = col
            df = pd.DataFrame(out)
        
        # 2. Type enforcement & validation
        df = enforce_types(df, self.aliases)
        validate_frame(df, self.aliases)
        return df