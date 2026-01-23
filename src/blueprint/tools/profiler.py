import pandas
import json
import logging
import pandas as pd

from pathlib import Path
from typing import Dict, Any

log = logging.getLogger(__name__)

def infer_column_spec(series: pd.Series) -> Dict[str, Any]:
    """
    Analyzes a pandas Series and returns 
    """
    spec = {}

    # 1. Null check
    if series.notna().all():
        spec["not_null"] = True
    
    # 2. Uniqueness check (Identifier candidate)
    # Only useful if dataset is reasonably large to avoid false positives on small samples
    if len(series) > 1 and series.is_unique:
        spec["identifier"] = True
    
    # 3. Type Inference
    try:
        numeric_series = pd.to_numeric(series.dropna(), errors='raise')
        # Check if integer
        if (numeric_series % 1 == 0).all():
            spec["type"] = "integer"
        else:
            spec["type"] = "number"
    except (ValueError, TypeError):
        # Not numeric.
        try:
            # Only try date if it looks like a string or object
            pd.to_datetime(series.dropna(), errors='raise')
            spec["type"] = "date"
            spec["date"] = { "format": "%Y-%m-%d" }
        except (ValueError, TypeError):
            # Fallback to string
            spec["type"] = "string"

            # 4. Enum Detection (Only for strings)
            # Heuristic: distinct values < 20 and distinct ratio < 10%
            distinct_count = series.nunique()
            if distinct_count > 0 and distinct_count < 20 and (distinct_count / len(series) < 0.1):
                # Sort for deterministic output
                unique_vals = sorted(series.dropna().unique().tolist())
                spec["enum"] = unique_vals
    return spec

def profile_file(path: str, sheet_name: str | int = 0) -> Dict[str, Any]:
    """
    Reads a file and generates a full 'schema' and 'sources' config block.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    # Load Data (sample first 10k rows to be safe/fast)
    if p.suffix in (".xlsx", ".xls", ".xlsb"):
        df = pd.read_excel(p, sheet_name=sheet_name, nrows=10000)
    elif p.suffix == ".csv":
        df = pd.read_csv(p, nrows=10000)
    else:
        raise ValueError("Unsupported file type for profiling.")

    # Generate Schema Aliases
    aliases = {}
    source_columns = {}

    for col in df.columns:
        # Create a snake_case alias
        clean_name = col.strip().lower().replace(" ", "_").replace("-", "_")
        clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")

        spec = infer_column_spec(df[col])
        aliases[clean_name] = spec

        # Source mapping
        source_columns[col] = { "alias": clean_name }
    
    # Construct the Config Objects
    generated_schema = {
        "aliases": aliases
    }

    generated_source = {
        "id": f"src_{p.stem}",
        "path": str(p),
        "tables": [
            {
                "name": str(sheet_name) if isinstance(sheet_name, int) else sheet_name,
                "table_id": f"tbl_{p.stem}",
                "columns": source_columns
            }
        ]
    }

    return {
        "schema": generated_schema,
        "source_snippet": generated_source
    }

if __name__ == "__main__":
    # Quick CLI for testing
    import sys
    if len(sys.argv) < 2:
        print("Usage: python profiler.py <path_to_excel>")
        sys.exit(1)

    res = profile_file(sys.argv[1])
    print(json.dumps(res, indent=2))