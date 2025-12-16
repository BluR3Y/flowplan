import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from flowplan.core.models import Config
from flowplan.io.adapters import get_adapter
from flowplan.ops.enrich import enrich_join, ENRICH_PLUGINS
from flowplan.engine.export import Exporter
from flowplan.engine.compare import write_compare_workbook
from flowplan.engine.expressions import build_mask

log = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Compilation Helpers
# -------------------------------------------------------------------------

def _apply_filters(per_table_frames: dict[str, pd.DataFrame], pre_filter_cfg: Dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    if not pre_filter_cfg:
        return per_table_frames
    out: dict[str, pd.DataFrame] = {}
    for table_id, df in per_table_frames.items():
        expr = pre_filter_cfg.get(table_id)
        if expr:
            mask = build_mask(df, expr)
            kept = int(mask.sum())
            log.info(f"[pre_filter] {table_id}: kept {kept}/{len(df)} rows")
            out[table_id] = df[mask].copy()
        else:
            out[table_id] = df
    return out

def _merge_inputs(inputs: list[str], frames: dict[str, pd.DataFrame], key: list[str]) -> pd.DataFrame:
    merged = None
    for i, src in enumerate(inputs):
        df = frames[src]
        if merged is None:
            merged = df.copy()
        else:
            # Outer merge to keep all records
            merged = merged.merge(df, how="outer", on=key, suffixes=("", f"__{i}"))
    return merged if merged is not None else pd.DataFrame()

def _apply_merge_rules(df: pd.DataFrame, rules: Dict[str, Any], inputs: list[str]) -> pd.DataFrame:
    if not rules:
        return df
    out = df.copy()
    for field, rule in rules.items():
        # A. Dict Strategy
        if isinstance(rule, dict):
            strat = rule.get("strategy")
            if strat == "first_non_null":
                priority = rule.get("priority") or []
                # Build list of columns: main field + suffixed fields for priority sources
                cols = [field] + [f"{field}__{inputs.index(src)}" for src in inputs if src in priority]
                # Filter to only existing columns
                cols = [c for c in cols if c in df.columns]
                if cols:
                    out[field] = df[cols].bfill(axis=1).iloc[:, 0]
            elif strat == "prefer_source":
                src = rule.get("prefer_source")
                cols = [field] + [c for c in df.columns if c.startswith(field + "__")]
                # Pick specific source col or fall back to first available
                pick = next((c for c in cols if src and src in c), cols[0] if cols else None)
                if pick:
                    out[field] = df[pick]
            continue

        # B. String Shorthand
        if isinstance(rule, str) and rule == "first_non_null":
            # Grab all columns starting with field + potential suffix
            candidates = [c for c in df.columns if c == field or c.startswith(f"{field}__")]
            if candidates:
                out[field] = df[candidates].bfill(axis=1).iloc[:, 0]
            continue

    # Cleanup: drop suffixed intermediate columns
    keep = [c for c in out.columns if "__" not in c]
    return out[keep]

def _apply_enrich(df: pd.DataFrame, frames: dict[str, pd.DataFrame], steps: list[dict]) -> pd.DataFrame:
    out = df.copy()
    for idx, step in enumerate(steps or []):
        try:
            if "from" in step and "left_on" in step and "right_on" in step:
                src = step["from"]
                if src not in frames:
                    raise ValueError(f"Enrich source '{src}' not found. Available: {list(frames.keys())}")
                dim = frames[src]
                out = enrich_join(
                    out, dim,
                    left_on=step["left_on"],
                    right_on=step["right_on"],
                    add=step.get("add", {}),
                    how=step.get("how", "left"),
                    on_conflict=step.get("on_conflict", "error"),
                    match=step.get("match"),
                )
            elif "fn" in step:
                fn = ENRICH_PLUGINS.get(step["fn"])
                if fn is None:
                    raise ValueError(f"Unknown enrich fn: {step['fn']}")
                out = fn(out, step.get("params", {}), frames=frames)
            else:
                raise ValueError(f"Invalid enrich step: {step}")
        except Exception as e:
            raise RuntimeError(f"[enrich step #{idx}] failed with error: {e}") from e
    return out

def _coerce_dtypes(df: pd.DataFrame, schema_aliases: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
    for col in columns:
        spec = (schema_aliases or {}).get(col) or {}
        t = spec.get("type")
        if not t:
            continue
        if t == "integer":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif t in ("number", "float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t == "boolean":
            df[col] = df[col].astype("boolean")
        elif t == "date":
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = df[col].astype("string")
    return df

def _assert_columns(df: pd.DataFrame, required: List[str], table_id: str):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[union] Source '{table_id}' missing required columns {missing}")

def _validate_keys(df: pd.DataFrame, keys: List[str], name: str):
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise ValueError(f"[diff] '{name}' missing key columns {missing}")

# -------------------------------------------------------------------------
# Compiler Class
# -------------------------------------------------------------------------

class Compiler:
    def __init__(self, frames: dict[str, pd.DataFrame], schema_aliases: Dict[str, Any] | None = None):
        self.frames = frames              # raw source frames
        self.compiled: dict[str, pd.DataFrame] = {}
        self.schema_aliases = schema_aliases or {}

    def compile_target(self, cfg: Dict[str, Any]) -> pd.DataFrame:
        name   = cfg.get("name")
        inputs = cfg.get("inputs") or []
        mode   = cfg.get("mode", "merge")
        pre_f  = cfg.get("pre_filter") or {}

        log.info(f"Compiling target: {name} (mode={mode})")

        # Validate inputs upfront
        missing = [i for i in inputs if i not in self.frames]
        if missing:
            raise ValueError(f"Target '{name}' references unknown inputs: {missing}. "
                             f"Available: {list(self.frames.keys())}")

        # Apply pre-filter per input
        per_table = {i: self.frames[i] for i in inputs}
        pre_filtered = _apply_filters(per_table, pre_f)

        # Route by mode
        if mode == "union":
            out = self._compile_union(cfg, pre_filtered)
        elif mode == "diff":
            out = self._compile_diff(cfg, pre_filtered)
        else:
            out = self._compile_merge(cfg, pre_filtered)
        
        self.compiled[name] = out
        return out

    def _compile_union(self, cfg: Dict[str, Any], pre_filtered_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        name     = cfg["name"]
        inputs   = cfg.get("inputs") or []
        columns  = cfg.get("columns")          
        add_src  = bool(cfg.get("add_source", False))
        dedupe   = cfg.get("dedupe_on")
        keep     = cfg.get("keep", "first")
        enrich_steps = cfg.get("enrich") or []
        post_f   = cfg.get("post_filter") or {}
        enrich_at = cfg.get("enrich_at", "post_dedupe")

        # Determine column set (intersection if not provided)
        if columns is None:
            cols_sets = [set(pre_filtered_frames[i].columns) for i in inputs]
            columns = sorted(set.intersection(*cols_sets)) if cols_sets else []
            if not columns:
                raise ValueError(f"[union] No common columns across inputs {inputs}")

        pieces = []
        for src in inputs:
            df = pre_filtered_frames[src].copy()
            _assert_columns(df, columns, src)
            df = df.reindex(columns=columns)
            df = _coerce_dtypes(df, self.schema_aliases, columns)
            if add_src:
                df["_source"] = src
            pieces.append(df)
            log.info(f"[union] '{src}' contributes {len(df)} rows")

        out = pd.concat(pieces, ignore_index=True, sort=False)

        # Dedupe Phase 1
        if dedupe and enrich_at == "post_dedupe":
            before = len(out)
            out = out.drop_duplicates(subset=dedupe, keep=keep)
            log.info(f"[union] dedup: {before} -> {len(out)}")
        
        # Enrich
        if enrich_steps:
            # Enrich can reference raw frames OR already compiled ones
            context = {**self.frames, **self.compiled}
            out = _apply_enrich(out, context, enrich_steps)

        # Dedupe Phase 2
        if dedupe and enrich_at == "pre_dedupe":
            before = len(out)
            out = out.drop_duplicates(subset=dedupe, keep=keep)
            log.info(f"[union] dedup (post-enrich): {before} -> {len(out)}")

        # Post-Filter
        if post_f:
            mask = build_mask(out, post_f)
            out = out[mask].copy()

        return out

    def _compile_merge(self, cfg: Dict[str, Any], pre_filtered_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        inputs = cfg.get("inputs") or []
        key = cfg.get("key") or []
        merge_rules = cfg.get("merge_rules") or {}
        enrich_steps = cfg.get("enrich") or []
        post_f = cfg.get("post_filter") or {}

        merged = _merge_inputs(inputs, pre_filtered_frames, key)
        merged = _apply_merge_rules(merged, merge_rules, inputs)

        if enrich_steps:
            context = {**self.frames, **self.compiled}
            merged = _apply_enrich(merged, context, enrich_steps)

        if post_f:
            mask = build_mask(merged, post_f)
            merged = merged[mask].copy()
        
        return merged

    def _compile_diff(self, cfg: Dict[str, Any], pre_filtered_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
        name    = cfg["name"]
        left_id = cfg["left"]
        right_id= cfg["right"]
        keys    = cfg.get("on") or []
        side    = cfg.get("side", "left_only")
        project = cfg.get("project", None)
        post_f  = cfg.get("post_filter") or {}
        add_src = bool(cfg.get("add_source", side == "symmetric_diff"))
        columns_left  = cfg.get("columns_left")
        columns_right = cfg.get("columns_right")

        left  = pre_filtered_frames[left_id].copy()
        right = pre_filtered_frames[right_id].copy()

        _validate_keys(left,  keys, left_id)
        _validate_keys(right, keys, right_id)

        # Optional Projection
        if columns_left is not None:
            left = left[keys + [c for c in columns_left if c not in keys]]
        if columns_right is not None:
            right = right[keys + [c for c in columns_right if c not in keys]]

        # Outer Merge
        merged = left.merge(
            right,
            how="outer",
            on=keys,
            suffixes=("_left", "_right"),
            indicator=True,
        )

        # Side Selection
        if side == "left_only":
            out = merged[merged["_merge"] == "left_only"].copy()
        elif side == "right_only":
            out = merged[merged["_merge"] == "right_only"].copy()
        elif side == "symmetric_diff":
            out = merged[merged["_merge"] != "both"].copy()
        elif side == "both":
            out = merged[merged["_merge"] == "both"].copy()
        else:
            raise ValueError(f"[diff] Invalid 'side': {side}")

        # Column Projection Logic
        if project is None:
            project = (
                "left" if side == "left_only" else
                "right" if side == "right_only" else
                "both"
            )

        key_cols = list(keys)
        final_cols = []
        
        if project == "keys":
            final_cols = key_cols
        elif project == "left":
            left_cols_sfx = [c for c in out.columns if c.endswith("_left")]
            out.rename(columns={c: c[:-5] for c in left_cols_sfx}, inplace=True)
            final_cols = key_cols + [c[:-5] for c in left_cols_sfx]
        elif project == "right":
            right_cols_sfx = [c for c in out.columns if c.endswith("_right")]
            out.rename(columns={c: c[:-6] for c in right_cols_sfx}, inplace=True)
            final_cols = key_cols + [c[:-6] for c in right_cols_sfx]
        elif project == "both":
            left_cols_sfx  = [c for c in out.columns if c.endswith("_left")]
            right_cols_sfx = [c for c in out.columns if c.endswith("_right")]
            final_cols = key_cols + left_cols_sfx + right_cols_sfx
        else:
            raise ValueError(f"[diff] Invalid 'project': {project}")

        # Add Source Column
        if add_src:
            out["_source"] = out["_merge"].map({
                "left_only": "left", "right_only": "right", "both": "both"
            })
            final_cols.append("_source")

        # Select Columns & Filter
        # Keep _merge if "both" was projected or generally useful, but typically we just clean up
        if "_merge" in out.columns and project == "both":
            final_cols.append("_merge")
            
        out = out[final_cols].copy()
        
        if post_f:
            mask = build_mask(out, post_f)
            out = out[mask].copy()

        return out

# -------------------------------------------------------------------------
# Pipeline Orchestrator
# -------------------------------------------------------------------------

class Pipeline:
    def __init__(self, config: Config):
        self.cfg = config
        self.frames: Dict[str, pd.DataFrame] = {}
        self.compiled: Dict[str, pd.DataFrame] = {}

    def load(self):
        """Load all raw sources into memory."""
        log.info("Starting Source Load...")
        for src in self.cfg.sources:
            adapter = get_adapter(src, self.cfg.schema_aliases)
            self.frames.update(adapter.load_tables())
        log.info(f"Loaded {len(self.frames)} source frames.")

    def compile(self):
        """Run the compilation targets (merge/union/diff)."""
        log.info("Starting Compilation...")
        compiler = Compiler(self.frames, self.cfg.schema_aliases)
        for tgt in self.cfg.compile_targets:
            self.compiled[tgt["name"]] = compiler.compile_target(tgt)

    def run(self):
        """Execute full pipeline: Load -> Compile -> Compare -> Export."""
        # 1. Load & Compile
        self.load()
        self.compile()

        # 2. Compare
        if self.cfg.compare_pairs:
            log.info(f"Starting Comparisons ({len(self.cfg.compare_pairs)} pairs)...")
            for pair in self.cfg.compare_pairs:
                left_key = pair["left"]
                right_key = pair["right"]
                
                if left_key not in self.compiled or right_key not in self.compiled:
                    log.error(f"Cannot compare {left_key} vs {right_key}: Dataset not found in compiled outputs.")
                    continue

                # Resolve output path: if not absolute, place in output_dir
                path = pair.get("path")
                if not path:
                    save_name = pair.get("save_name", f"{left_key}_vs_{right_key}")
                    path = str(Path(self.cfg.output or "out") / f"{save_name}.xlsx")
                
                write_compare_workbook(
                    self.compiled[left_key],
                    self.compiled[right_key],
                    key_cols=pair.get("on"),
                    compare_cols=pair.get("compare_cols") or list(self.compiled[left_key].columns),
                    path=path,
                    # Pass extra options if present in config pair
                    id_label=pair.get("id_label", "Baseline"),
                    new_label=pair.get("new_label", "Current")
                )

        # 3. Export
        if self.cfg.export_workbooks:
            log.info(f"Starting Exports ({len(self.cfg.export_workbooks)} workbooks)...")
            exporter = Exporter(self.cfg.output)
            paths = []
            for wb in self.cfg.export_workbooks:
                paths.append(exporter.export_workbook(wb, self.compiled))
            log.info(f"Exported: {paths}")