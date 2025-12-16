from __future__ import annotations
from typing import (
    Any, Dict, Iterable, List, Mapping,
    Optional, Tuple, Literal, Callable, Union
)

import pandas as pd
import numpy as np
import logging

log = logging.getLogger(__name__)

class DocumentManager:
    # Define flexible input types
    DataType = Union[pd.DataFrame, List[Mapping[str, Any]], Mapping[str, List[Any]]]

    def __init__(
        self,
        name: str,
        data: DataType,
        on_change: Optional[Callable[[], None]] = None
    ) -> None:
        self.name = name
        self.df = self._normalize_data(data)
        self._on_change = on_change
    
    def _normalize_data(self, data: DataType) -> pd.DataFrame:
        """
        Converts various input formats (list of dicts, dict of lists) into a DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            return data
        
        try:
            # Pandas handles List[dict] and Dict[list] automatically
            return pd.DataFrame(data)
        except Exception as e:
            log.error(f"Failed to convert data for document '{self.name}': {e}")
            raise ValueError(f"Invalid data format for document '{self.name}': {e}") from e
    
    def _notify_change(self) -> None:
        """Trigger the dirty flag in the parent manager."""
        if self._on_change:
            self._on_change()
    
    # --------------------------- Data Access ---------------------------
    # Altered from original get_df fn
    def get_df(self) -> pd.DataFrame:
        """
        Return the internal DataFrame.
        """
        return self.df
    
    # --------------------------- Mutations ---------------------------
    def update_row(self, row: int, cols: Mapping[str, Any]) -> None:
        """Update specified cells in row by integer index."""
        # 1. Bounds Check
        n_rows = len(self.df)
        if not (0 <= row < n_rows):
            raise IndexError(f"Row {row} is out of bounds (Size: {n_rows}).")

        # 2. Resolve Column Indices (Fail fast if column missing)
        try:
            # Create a list of integer locations for the column names
            col_indices = [self.df.columns.get_loc(c) for c in cols.keys()]
        except KeyError as e:
            raise KeyError(f"Column '{e.args[0]}' not found in collection '{self.name}'.")

        # 3. Update (Atomic operation)
        # using list(cols.values()) ensures order matches cols.keys()
        self.df.iloc[row, col_indices] = list(cols.values())
        
        self._notify_change()

    def append_rows(self, rows: List[Mapping[str, Any]]) -> None:
        """
        Batch append rows to the dataframe.
        Time Complexity: O(N)
        """
        if not rows:
            return
        
        new_data = pd.DataFrame(rows)
        # Align columns to ensure consistency
        if not new_data.empty:
            # Use concat for efficient memory allocation
            self.df = pd.concat([self.df, new_data], ignore_index=True)
            self._notify_change()
    
    def delete_row(self, row: int) -> None:
        self.df.drop(index=row, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        self._notify_change()

    # --------------------------- Search Logic ---------------------------
    def match(
        self,
        query: Mapping[str, Any],
        *,
        candidates: Optional[pd.DataFrame] = None,
        threshold: float = 0.6,
        include_score: bool = True,
        best_only: bool = False,
        nulls_match: bool = False,
    ) -> pd.DataFrame:
        """
        Fuzzy or Exact match rows based on query dict.
        """
        if not query:
            return self.df.iloc[0:0].copy()

        df = candidates if candidates is not None else self.df
        if df.empty:
            return df.iloc[0:0].copy()

        # Filter keys to only those present in columns
        keys = [k for k in query.keys() if k in df.columns]
        if not keys:
            return df.iloc[0:0].copy()

        # Exact Match Optimization
        if threshold >= 1.0:
            mask = pd.Series(True, index=df.index)
            for k in keys:
                mask &= (df[k] == query[k])
            
            out = df[mask].copy()
            if include_score:
                out["__score__"] = 1.0
            return out.iloc[[0]] if best_only and not out.empty else out

        # Vectorized Fuzzy Match
        # Create a query series for broadcasting
        qser = pd.Series({k: query[k] for k in keys})
        
        # Compare all keys at once
        comp = df[keys].eq(qser)

        if nulls_match:
            # If both are NaN/None, count as match
            comp |= (df[keys].isna() & pd.isna(qser))

        # Calculate score (mean of matches)
        scores = comp.sum(axis=1) / len(keys)
        
        keep = scores >= threshold
        out = df.loc[keep].copy()
        
        if include_score:
            out["__score__"] = scores[keep].astype(float)
            out.sort_values("__score__", ascending=False, inplace=True)
            
        return out.iloc[[0]] if best_only and not out.empty else out