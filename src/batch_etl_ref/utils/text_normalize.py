import re
import unicodedata
import pandas as pd
from typing import List

# --- Text Normalization ---
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")

def normalize_text(s: str | None, steps: List[str]) -> str | None:
    if s is None:
        return None
    out = s
    if "nfkc" in steps:
        out = unicodedata.normalize("NFKC", out)
    if "strip" in steps:
        out = out.strip()
    if "collapse_ws" in steps:
        out = _WS_RE.sub(" ", out)
    if "strip_punct" in steps:
        out = _PUNCT_RE.sub("", out)
    if "lower" in steps:
        out = out.lower()
    elif "upper" in steps:
        out = out.upper()
    elif "title" in steps:
        out = out.title()
    return out

def normalize_series(series: pd.Series, steps: List[str]) -> pd.Series:
    """Vectorized normalization for a series."""
    # Ensure string type
    s = series.astype("string")

    # Pre-compile steps for performance where possible
    if "nfkc" in steps:
        s = s.map(lambda x: unicodedata.normalize("NFKC", x) if pd.notna(x) else x)
    if "strip_punct" in steps:
        s = s.str.replace(r"[^\w\s]", "", regex=True)
    if "collapse_ws" in steps:
        s = s.str.replace(r"\s+", " ", regex=True)
    if "strip" in steps:
        s = s.str.strip()
    if "lower" in steps:
        s = s.str.lower()
    elif "upper" in steps:
        s = s.str.upper()
    elif "title" in steps:
        s = s.str.title()
    
    return s