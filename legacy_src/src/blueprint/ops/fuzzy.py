import pandas as pd
import numpy as np
import unicodedata
import re
from typing import List, Tuple, Dict, Optional
from rapidfuzz import process, fuzz

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SCORERS = {
    "ratio": fuzz.ratio,
    "partial_ratio": fuzz.partial_ratio,
    "token_sort_ratio": fuzz.token_sort_ratio,
    "token_set_ratio": fuzz.token_set_ratio,
}

def normalize_str(s: str | None, steps: list[str]) -> str | None:
    if s is None: return None
    out = str(s)
    if "nfkc" in steps: out = unicodedata.normalize("NFKC", out)
    if "strip" in steps: out = out.strip()
    if "lower" in steps: out = out.lower()
    if "collapse_ws" in steps: out = _WS_RE.sub(" ", out)
    if "strip_punct" in steps: out = _PUNCT_RE.sub("", out)
    return out

def _block_key(s: str | None, mode: str) -> Optional[str]:
    if not s: return None
    if mode == "first_char": return s[:1]
    if mode == "first2": return s[:2]
    return None

def fuzzy_match_series(
    left: pd.Series,
    right_unique: pd.Series,
    *,
    normalize_steps: List[str],
    scorer: str,
    threshold: int,
    top_k: int,
    block: str = "first_char",
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Optimized fuzzy matcher using unique value iteration.
    Returns (match_to, match_score, match_method) aligned with left.index.
    """
    # 1. Prepare Right Side
    r_orig = right_unique.dropna().astype(str)
    r_norm = r_orig.map(lambda v: normalize_str(v, normalize_steps) or "")
    
    # Map normalized -> original (first wins)
    norm_to_orig: Dict[str, str] = {}
    for normed, orig in zip(r_norm, r_orig):
        norm_to_orig.setdefault(normed, orig)
    
    # Buckets for right side
    candidates_by_block: Dict[str, List[str]] = {}
    for normed in norm_to_orig.keys():
        b = _block_key(normed, block) or ""
        candidates_by_block.setdefault(b, []).append(normed)

    all_norm_candidates = list(norm_to_orig.keys())
    scorer_fn = _SCORERS.get(scorer, fuzz.token_sort_ratio)

    # 2. Process Unique Left Values
    l_orig_unique = left.dropna().astype(str).unique()
    
    # Store results: input_val -> (result_val, score, method)
    # Method codes: 0=miss, 1=exact, 2=normalized, 3=fuzzy
    results_map: Dict[str, Tuple[Optional[str], int, str]] = {}
    
    # Set for O(1) exact lookups
    r_orig_set = set(r_orig.values)

    for val in l_orig_unique:
        s_norm = normalize_str(val, normalize_steps) or ""
        
        # A. Exact
        if val in r_orig_set:
            results_map[val] = (val, 100, "exact")
            continue
            
        # B. Normalized Exact
        if s_norm in norm_to_orig:
            results_map[val] = (norm_to_orig[s_norm], 100, "normalized")
            continue
            
        # C. Fuzzy
        block_val = _block_key(s_norm, block) or ""
        cands = candidates_by_block.get(block_val) or all_norm_candidates
        
        best = process.extractOne(s_norm, cands, scorer=scorer_fn, score_cutoff=threshold)
        
        if best:
            best_norm, score, _ = best
            results_map[val] = (norm_to_orig[best_norm], int(score), "fuzzy")
        else:
            results_map[val] = (None, 0, "miss")

    # 3. Broadcast results back to original series
    # Using map() is much faster than iterating rows
    
    def get_res(x):
        if pd.isna(x): return (None, 0, "miss")
        return results_map.get(str(x), (None, 0, "miss"))
    
    # Create an intermediate series of tuples
    mapped = left.map(get_res)
    
    # Unpack into three series
    # Note: Using .tolist() + constructor is often faster than .apply(pd.Series)
    unpacked = mapped.tolist()
    
    out_to = pd.Series([u[0] for u in unpacked], index=left.index, dtype=object)
    out_score = pd.Series([u[1] for u in unpacked], index=left.index, dtype="Int64")
    out_method = pd.Series([u[2] for u in unpacked], index=left.index, dtype=object)
    
    return out_to, out_score, out_method