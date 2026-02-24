from .base_adapter import ExtractTask
import pandas as pd
import json
import logging

log = logging.getLogger(__name__)

@ExtractTask.register_adapter("json")
def _json_extract(path: str) -> dict[str, pd.DataFrame]:
    with open(path) as f:
        d = json.load(f)
        # Removing column filtering
        return d