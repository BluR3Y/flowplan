from __future__ import annotations
from typing import (
    Any, Dict, Iterable, List, Mapping,
    Optional, Tuple, Literal, Callable, Union
)

import pandas as pd
import numpy as np
import logging

from .document_manager import DocumentManager
log = logging.getLogger(__name__)


class SheetManager(DocumentManager):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        on_change: Optional[Callable[[], None]] = None
    ) -> None:
        super().__init__(name, df, on_change)
        self._annotations: List[Tuple[int, int, str, str]] = []

    # --------------------------- Annotations ---------------------------
    def add_annotation(
        self,
        row: int,
        col: Union[int, str],
        level: Literal["info", "warn", "error"],
        message: str
    ) -> None:
        """
        Record a visual annotation (comment/color) for a cell.
        Does Not trigger _notify_change by default (optional design choice),
        assuming annotations are derived metadata
        """
        if isinstance(col, str):
            try:
                col = self.df.columns.get_loc(col)
            except KeyError:
                raise ValueError(f"Column '{col}' not found.")
        
        self._annotations.append((row, col, level, message))
        self._on_change()
    
    def iter_annotations(self) -> Iterable[Tuple[int, int, str, str]]:
        return iter(self._annotations)