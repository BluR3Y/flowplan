from .base_adapter import SourceAdapter
import pandas as pd
import logging

log = logging.getLogger(__name__)

class AccessAdapter(SourceAdapter, type_id="access"):
    def load_data(self) -> dict[str, pd.DataFrame]:
        try:
            import pyodbc
        except ImportError:
            raise RuntimeError("pyodbc missing")

        path = self.cfg.get("path")
        conn_str = f"Driver={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={path};"
        cn = pyodbc.connect(conn_str)
        data = {}
        for name, specs in self.cfg.get("data", {}).items():
            id = specs.get("dataset_id") or name
            log.info(f"Loading Access table '{name}' from {path}")
            df = pd.read_sql(f"SELECT * FROM [{name}]", cn)
            data[id] = self._process_data(df, specs)
        cn.close()
        return data