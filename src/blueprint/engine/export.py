import os
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE

from blueprint.engine.expressions import eval_expr, build_mask
from blueprint.ops.transforms import apply_pipeline
from blueprint.exceptions import ExportError

log = logging.getLogger(__name__)

def _resolve_output_dir(config_out: Optional[str]) -> Path:
    """
    Resolves the output directory path. 
    Defaults to './out' if not specified.
    """
    if not config_out:
        return Path.cwd() / "out"
    # os.path.expandvars handles environment variables like $HOME or %USERPROFILE%
    # that might have been passed strictly as strings outside the config loader.
    expanded = os.path.expandvars(config_out)
    return Path(expanded)

class Exporter:
    def __init__(self, output_dir: Optional[str]):
        self.out_dir = _resolve_output_dir(output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_excel_strings(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean strings for Excel:
          - strip Excel-illegal control chars (like null bytes)
          - normalize newlines
          - clip to 32,767 chars (Excel cell limit)
        Works safely with NA values.
        """
        out = df.copy()
        pattern = ILLEGAL_CHARACTERS_RE.pattern

        for col in out.columns:
            # Only process object/string columns
            if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
                # Convert to nullable string type to handle NAs gracefully
                s = out[col].astype("string")

                # 1) strip illegal control chars
                s = s.str.replace(pattern, " ", regex=True)

                # 2) normalize newlines
                s = s.str.replace("\r\n", "\n", regex=False).str.replace("\r", "\n", regex=False)

                # 3) clip to Excel cell limit
                s = s.str.slice(0, 32767)

                out[col] = s
        return out

    def export_workbook(self, wb_cfg: Dict[str, Any], compiled: Dict[str, pd.DataFrame]) -> str:
        """
        Generates an Excel workbook based on the provided configuration.
        """
        save_name = wb_cfg.get("save_name", "export") + ".xlsx"
        out_path = self.out_dir / save_name
        log.info(f"Exporting workbook: {out_path}")

        created_any = False

        # engine="openpyxl" is standard for .xlsx writing
        with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
            for sheet in wb_cfg.get("sheets", []):
                name = sheet.get("name", "Sheet1")
                filt = sheet.get("filter") or {}
                columns = sheet.get("columns", {}) or {}
                source_name = sheet.get("from")

                # Validate source existence
                if source_name not in compiled:
                    # If source isn't found, try to default to the first compiled frame if available
                    # otherwise skip or raise error. Here we skip with a warning.
                    if not compiled:
                        log.warning(f"Sheet '{name}' skipped: No compiled data available.")
                        continue
                    if source_name is None:
                        source_name = next(iter(compiled.keys()))
                    else:
                        log.warning(f"Sheet '{name}' skipped: Source '{source_name}' not found.")
                        continue

                try:
                    df = compiled[source_name].copy()

                    # 1. Apply Row Filters
                    if filt:
                        mask = build_mask(df, filt)
                        df = df[mask].copy()

                    # 2. Build Output Columns
                    out_map: Dict[str, pd.Series] = {}
                    
                    for header, spec in columns.items():
                        # Normalize spec to dict if it's just a string alias
                        spec = spec if isinstance(spec, dict) else {"alias": str(spec)}
                        series = None

                        # A) Determine Base Data
                        if "value" in spec:
                            # broadcast scalar constant
                            series = pd.Series([spec["value"]] * len(df), index=df.index)
                        elif "compute" in spec:
                            # Evaluate AST expression
                            series = eval_expr(df, spec["compute"])
                        elif "alias" in spec:
                            # Lookup existing column
                            col_name = spec["alias"]
                            if col_name in df.columns:
                                series = df[col_name]
                            else:
                                # Start with None/NaN if alias missing
                                series = pd.Series([None] * len(df), index=df.index)
                        else:
                            # Fallback if config is malformed
                            series = pd.Series([None] * len(df), index=df.index)

                        # B) Apply Column Transforms (optional)
                        transforms = spec.get("transforms", [])
                        if transforms and series is not None:
                            series = apply_pipeline(series, transforms)

                        out_map[header] = series

                    out_df = pd.DataFrame(out_map, index=df.index)

                    # 3. Final Cleanup
                    if out_df.empty:
                        log.warning(f"Sheet '{name}' produced 0 rows; skipping.")
                        continue
                    
                    # Normalize NAs to None (helps openpyxl handle mixed dtypes better)
                    out_df = out_df.where(pd.notna(out_df), None)

                    # Sanitize strings
                    out_df = self._sanitize_excel_strings(out_df)

                    # Write Sheet
                    out_df.to_excel(xl, sheet_name=name, index=False)
                    created_any = True

                except Exception as e:
                    log.exception(f"Sheet '{name}' failed during export: {e}. Skipping.")
                    continue

        # Ensure valid file creation if no sheets were written
        if not created_any:
            with pd.ExcelWriter(out_path, engine="openpyxl") as xl:
                pd.DataFrame({"info": ["no data exported"]}).to_excel(xl, sheet_name="README", index=False)
            log.warning("No valid sheets generated; emitted placeholder README.")

        return str(out_path)