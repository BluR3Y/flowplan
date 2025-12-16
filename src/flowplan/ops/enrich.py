import pandas as pd
import numpy as np
import logging
from typing import Mapping, Optional, Union, Tuple, TypedDict, List, Literal, Any

from flowplan.ops.fuzzy import fuzzy_match_series
from flowplan.utils import load_entrypoints

log = logging.getLogger(__name__)

ENRICH_PLUGINS = load_entrypoints("flowplan.enrich")

# --- Configuration TypedDicts ---

class FuzzyCfg(TypedDict, total=False):
    scorer: Literal["ratio", "partial_ratio", "token_sort_ratio", "token_set_ratio"]
    threshold: int
    top_k: int
    block: str

class MatchCfg(TypedDict, total=False):
    strategy: List[Literal["exact", "normalized", "fuzzy"]]
    normalize: List[str]
    fuzzy: FuzzyCfg
    left_kind: Literal["string", "integer", "number", "date"]
    right_kind: Literal["string", "integer", "number", "date"]
    cardinality: Literal["m:1", "1:1"]
    min_len: int
    right_keep: Literal["first", "last"]
    on_miss: Literal["leave_null", "fail"]
    write_policy: Literal["fillna", "overwrite"]

# --- Helpers ---

def _nullable_int_dtype(dtype: Any) -> Any:
    """Normalize int dtypes to pandas nullable Int64."""
    if pd.api.types.is_integer_dtype(dtype):
        return "Int64"
    return dtype

def _coerce_kind(series: pd.Series, kind: str, for_text_ops: bool) -> pd.Series:
    """Coerce to appropriate dtype for matching."""
    kind = (kind or "string").lower()
    if for_text_ops:
        return series.astype("string")
    if kind in ("integer", "number"):
        return pd.to_numeric(series, errors="coerce")
    if kind == "date":
        return pd.to_datetime(series, errors="coerce")
    return series.astype("string")

def _assert_cardinality(right_keys: pd.Series, cardinality: str):
    if cardinality == "1:1":
        counts = right_keys.value_counts(dropna=True)
        dups = counts[counts > 1]
        if not dups.empty:
            raise ValueError(
                f"enrich.cardinality=1:1 but right side has duplicate keys. "
                f"Count: {len(dups)}. Examples: {list(map(str, dups.index[:3]))}"
            )

# --- Main Logic ---

def enrich_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_on: str,
    right_on: str,
    add: Mapping[str, str],
    how: Literal["left", "inner"] = "left",
    match: Optional[MatchCfg] = None,
    on_conflict: Literal["error", "skip", "overwrite", "suffix"] = "error",
    return_audit: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Enriches 'left' DataFrame with columns from 'right' using fuzzy/exact matching logic.
    """
    # 1. Configuration Defaults
    cfg: MatchCfg = {
        "strategy": ["exact"],
        "normalize": ["strip", "collapse_ws", "strip_punct", "nfkc", "strip_accents", "lower"],
        "fuzzy": {"scorer": "token_set_ratio", "threshold": 90, "top_k": 1, "block": "first_char"},
        "left_kind": "string",
        "right_kind": "string",
        "cardinality": "m:1",
        "right_keep": "first",
        "on_miss": "leave_null",
        "write_policy": "fillna",
        **(match or {}),
    }
    
    strategy = [s.lower() for s in cfg.get("strategy", ["exact"])]
    norm_steps = cfg.get("normalize", [])
    fuzzy_cfg = cfg.get("fuzzy", {})
    left_kind = cfg.get("left_kind", "string")
    right_kind = cfg.get("right_kind", "string")
    right_keep = cfg.get("right_keep", "first")
    write_policy = cfg.get("write_policy", "fillna")
    
    # 2. Validation
    if left_on not in left.columns:
        raise ValueError(f"Left key '{left_on}' not found")
    if right_on not in right.columns:
        raise ValueError(f"Right key '{right_on}' not found")
    
    # 3. Prepare Keys
    text_ops = ("normalized" in strategy) or ("fuzzy" in strategy)
    left_key = _coerce_kind(left[left_on], left_kind, text_ops)
    right_key = _coerce_kind(right[right_on], right_kind, text_ops)

    # 4. Prepare Right Side Lookup
    # We must deduplicate the right side based on 'right_keep' logic 
    # so we can map matched_key -> row_values deterministically.
    right_dedup = right.copy()
    right_dedup["__key__"] = right_key
    
    # Filter null keys
    right_dedup = right_dedup[right_dedup["__key__"].notna()]
    
    # Check cardinality before dedup if 1:1 requested
    if cfg.get("cardinality") == "1:1":
        _assert_cardinality(right_dedup["__key__"], "1:1")

    # Deduplicate right side for lookup
    right_dedup = right_dedup.drop_duplicates(subset=["__key__"], keep=right_keep)
    right_lookup_map = right_dedup.set_index("__key__")

    # 5. Run Matching Engine
    # This uses the optimized unique-value fuzzy matcher from ops/fuzzy.py
    matched_vals, scores, methods = fuzzy_match_series(
        left=left_key,
        right_unique=right_dedup["__key__"],
        normalize_steps=norm_steps,
        scorer=fuzzy_cfg.get("scorer", "token_set_ratio"),
        threshold=fuzzy_cfg.get("threshold", 90),
        top_k=fuzzy_cfg.get("top_k", 1),
        block=fuzzy_cfg.get("block", "first_char")
    )

    # 6. Apply Strategy Constraints
    # The engine tries everything; we filter back if the user didn't want fuzzy/norm.
    allowed_methods = set(strategy)
    mask_allowed = methods.isin(allowed_methods) | (methods == "exact") # Exact always allowed if found
    
    # If method not allowed, treat as miss
    matched_vals.loc[~mask_allowed] = None
    methods.loc[~mask_allowed] = "miss"
    scores.loc[~mask_allowed] = 0

    # 7. Merge Logic (Inner vs Left)
    out = left.copy()
    match_mask = matched_vals.notna()

    if how == "inner":
        out = out[match_mask].reset_index(drop=True)
        matched_vals = matched_vals[match_mask].reset_index(drop=True)
        scores = scores[match_mask].reset_index(drop=True)
        methods = methods[match_mask].reset_index(drop=True)
        match_mask = pd.Series([True] * len(out), index=out.index)

    # 8. Write Columns
    # We perform a map/lookup from the matched_val to the right_dedup row
    for target_col, src_col in add.items():
        if src_col not in right_lookup_map.columns:
            raise ValueError(f"Source column '{src_col}' not found in right table")

        # Get values from right table based on matched keys
        # map() handles the alignment to 'out' index
        new_values = matched_vals.map(right_lookup_map[src_col])
        
        # Conflict Handling
        final_target = target_col
        if target_col in out.columns:
            if on_conflict == "error":
                raise ValueError(f"Column '{target_col}' already exists")
            elif on_conflict == "skip":
                continue
            elif on_conflict == "suffix":
                k = 1
                while f"{target_col}_r{k}" in out.columns: k += 1
                final_target = f"{target_col}_r{k}"
            # else overwrite -> stay as target_col

        # Write Data
        r_dtype = _nullable_int_dtype(right[src_col].dtype)
        
        if final_target not in out.columns:
            out[final_target] = new_values.astype(r_dtype)
        else:
            if write_policy == "overwrite":
                out[final_target] = new_values.astype(r_dtype)
            else: # fillna
                out[final_target] = out[final_target].fillna(new_values)

    # 9. Handle On Miss
    if cfg.get("on_miss") == "fail" and how == "left":
        # Check for non-null left keys that failed to match
        missed = left_key.notna() & (left_key.str.strip() != "") & matched_vals.isna()
        if missed.any():
            sample = left.loc[missed, [left_on]].head(5).to_dict(orient="records")
            raise ValueError(f"enrich_join: failed to match rows. Examples: {sample}")

    # 10. Audit
    if return_audit:
        audit = pd.DataFrame({
            "left_idx": out.index,
            "left_key_raw": left_key if how == "left" else left_key[match_mask].reset_index(drop=True),
            "matched_val": matched_vals,
            "score": scores,
            "method": methods
        })
        # Filter to only matches for cleaner audit? Usually we want to see misses too.
        return out, audit

    return out