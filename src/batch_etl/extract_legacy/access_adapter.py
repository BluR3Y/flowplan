from .base_adapter import ExtractTask
import pandas as pd
from typing import Dict, List, Union, Any
import logging

log = logging.getLogger(__name__)

@ExtractTask.register_adapter("access")
def _access_extract(path: str, data: Dict[str, dict]) -> pd.DataFrame:
    try:
        import pyodbc
    except ImportError:
        raise RuntimeError("pyodbc package missing")
    
    conn_str = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path};"
    cn = pyodbc.connect(conn_str)

    # if isinstance(columns, Dict):
    #     selected_cols = ""
    # elif isinstance(columns, List):
    #     selected_cols = ",".join(columns)
    # else:
    #     selected_cols = "*"
    data = {}
    for name, config in data.items():
        