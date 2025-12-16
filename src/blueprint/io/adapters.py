import pandas as pd
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from blueprint.config.validator import enforce_types, validate_frame
from blueprint.ops.transforms import apply_pipeline

log = logging.getLogger(__name__)

class SourceAdapter(ABC):
    def __init__(self, source_cfg: Dict[str, Any], aliases: Dict[str, Any]):
        self.cfg = source_cfg
        self.aliases = aliases
    
    @abstractmethod
    def load_tables(self) -> dict[str, pd.DataFrame]: ...
    
    def _load_table_common(self, df: pd.DataFrame, table_cfg: Dict[str, Any]) -> pd.DataFrame:
        cols = table_cfg.get("columns", {})
        if cols:
            out = {}
            for src_col, spec in cols.items():
                alias = spec.get("alias") if isinstance(spec, dict) else None
                transforms = spec.get("transforms", []) if isinstance(spec, dict) else []
                col = df[src_col] if src_col in df.columns else pd.Series([None]*len(df))
                if transforms: col = apply_pipeline(col, transforms)
                out[alias or src_col] = col
            df = pd.DataFrame(out)
        
        df = enforce_types(df, self.aliases)
        validate_frame(df, self.aliases)
        return df

class ExcelAdapter(SourceAdapter):
    def load_tables(self) -> dict[str, pd.DataFrame]:
        path = self.cfg.get("path")
        tables = {}
        for t in self.cfg.get("tables", []):
            name = t.get("name")
            log.info(f"Loading Excel sheet '{name}'")
            df = pd.read_excel(path, sheet_name=name)
            tables[t.get("table_id") or name] = self._load_table_common(df, t)
        return tables

class AccessAdapter(SourceAdapter):
    def load_tables(self) -> dict[str, pd.DataFrame]:
        try:
            import pyodbc
        except ImportError:
            raise RuntimeError("pyodbc not installed. Cannot load Access DB.")
            
        path = self.cfg.get("path")
        conn_str = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path};"
        cn = pyodbc.connect(conn_str)
        tables = {}
        try:
            for t in self.cfg.get("tables", []):
                name = t.get("name")
                log.info(f"Loading Access table '{name}'")
                df = pd.read_sql(f"SELECT * FROM [{name}]", cn)
                tables[t.get("table_id") or name] = self._load_table_common(df, t)
        finally:
            cn.close()
        return tables

class InlineAdapter(SourceAdapter):
    def load_tables(self) -> dict[str, pd.DataFrame]:
        tables = {}
        for spec in self.cfg.get("data", []):
            df = pd.DataFrame(spec.get("rows") or [])
            tables[spec["table_id"]] = self._load_table_common(df, spec)
        return tables

def get_adapter(src: dict, aliases: dict) -> SourceAdapter:
    if "data" in src: return InlineAdapter(src, aliases)
    path = (src.get("path") or "").lower()
    if path.endswith((".xlsx", ".xls", ".xlsb")): return ExcelAdapter(src, aliases)
    if path.endswith((".accdb", ".mdb")): return AccessAdapter(src, aliases)
    raise ValueError(f"Unknown source type: {src}")