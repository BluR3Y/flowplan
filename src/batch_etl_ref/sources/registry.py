from typing import Dict, Type

from .base import SourceAdapter
from .excel import ExcelAdapter
from .access import AccessAdapter
from .json import JsonAdapter

# Registry Logic
_ADAPTERS: Dict[str, Type[SourceAdapter]] = {}

def register_adapter(key: str, cls: Type[SourceAdapter]):
    _ADAPTERS[key] = cls

# Register defaults
register_adapter("excel", ExcelAdapter)
register_adapter("access", AccessAdapter)
register_adapter("json", JsonAdapter)

def get_adapter(src_cfg: dict, aliases: dict) -> SourceAdapter:
    # Heuristic detection if not explicit type
    if "type" in src_cfg:
        typ = src_cfg["type"]
    elif src_cfg.get("path", "").lower().endswith((".json", ".jsonc")):
        typ = "json"
    elif src_cfg.get("path", "").lower().endswith((".xlsx", ".xls", ".xlsb")):
        typ = "excel"
    elif src_cfg.get("path", "").lower().endswith((".accdb", ".mdb")):
        typ = "access"
    else:
        raise ValueError(f"Cannot determine source type: {src_cfg}")
    
    cls = _ADAPTERS.get(typ)
    if not cls:
        raise ValueError(f"No adapter for source type: {typ}")
    return cls(src_cfg, aliases)