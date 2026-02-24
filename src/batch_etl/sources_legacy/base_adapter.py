from abc import ABC, abstractmethod
from typing import Dict, Any, Type
import pandas as pd

class SourceAdapter(ABC):
    _registry: Dict[str, Type["SourceAdapter"]] = {}

    def __init__(self, source_cfg: Dict[str, Any]):
        self.cfg = source_cfg

    def __init_subclass__(cls, type_id, **kwargs):
        super().__init_subclass__(**kwargs)
        if type_id is not None:
            if type_id in cls._registry:
                raise ValueError(f"Source type '{type_id}' already registered.")
            cls._registry[type_id] = cls
    
    @classmethod
    def extract(cls, source_cfg: Dict[str, Any]):
        path = source_cfg.get("path")
        # Heuristic adapter detection
        if path.lower().endswith(".json"):
            adapter = cls._registry.get("json")
        elif path.lower().endswith((".xlsx", ".xls", ".xlsb")):
            adapter = cls._registry.get("excel")
        elif path.lower().endswith((".accdb", ".mdb")):
            adapter = cls._registry.get("access")
        else:
            raise ValueError(f"Cannot determine source type: {path}")
        
        return adapter(source_cfg)

    @abstractmethod
    def load_data(self) -> dict[str, pd.DataFrame]:
        ...

    def _process_data(self, df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
        # Copying columns
        cols_list = data_cfg.get("columns", [])
        if cols_list:
            out = {}
            for col_name in cols_list:
                col = df[col_name] if col_name in df.columns else pd.Series([None]*len(df))
                out[col_name] = col
            df = pd.DataFrame(out)

        return df

    # def _process_data(self, df: pd.DataFrame, data_cfg: Dict[str, Any]) -> pd.DataFrame:
    #     # 1. Column mapping
    #     cols_map = data_cfg.get("columns", {})
    #     if cols_map:
    #         out = {}
    #         for col, spec in cols_map.items():
    #             alias = spec.get("alias") if isinstance(spec, dict) else None
    #             col = df[col] if col in df.columns else pd.Series([None]*len(df))
    #             out[alias or col] = col
    #         df = pd.DataFrame(out)
    #         # ** Removed pre-transform transformation

    #     # 2. Type enforcement & validation
    #     # df = enforce_types(df, self.aliases)
    #     # validate_frame(df, self.aliases)
    #     return df