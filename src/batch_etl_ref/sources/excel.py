from .base import SourceAdapter
import logging
import pandas as pd

log = logging.getLogger(__name__)

class ExcelAdapter(SourceAdapter):
    def load_data(self) -> dict[str, pd.DataFrame]:
        path = self.cfg.get("path")
        data = {}
        for name, specs in self.cfg.get("data", {}).items():
            id = specs.get("dataset_id") or name
            log.info(f"Loading Excel sheet '{name}' from {path}")
            df = pd.read_excel(path, sheet_name=name)
            data[id] = self._process_data(df, name)
        return data