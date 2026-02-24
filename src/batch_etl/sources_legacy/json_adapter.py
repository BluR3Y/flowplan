from .base_adapter import SourceAdapter
import pandas as pd
import json
import logging

log = logging.getLogger(__name__)

class JsonAdapter(SourceAdapter, type_id="json"):
    def load_data(self) -> dict[str, pd.DataFrame]:
        path = self.cfg.get("path")
        data = {}
        with open(path) as f:
            d = json.load(f)
            for name, specs in self.cfg.get("data", {}).items():
                id = specs.get("dataset_id") or name
                log.info(f"Loading JSON data '{name}' from {path}")
                df = pd.DataFrame(d.get(name, []))
                data[id] = self._process_data(df, specs)
        return data