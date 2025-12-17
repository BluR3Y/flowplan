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
    """
    Extends DocumentManager with Excel-specific features
    """
    def __init__(
        self,
        name: str,
        data: DocumentManager.DataType,
        on_change: Optional[Callable[[], None]] = None
    ) -> None:
        # Init: Let DocumentManager create the DataFrame
        super().__init__(name, data, on_change)

        # Normalize: Ensure strict Pandas types.
        # This converts generic 'objects' to proper types (e.g. string numbers to ints, lists of dates to Timestamps)
        # errors='ignore' ensures we don't crash on truly messy data
        self.df = self.df.convert_dtypes()

        self._annotations: List[Tuple[int, int, str, str]] = []

    def to_excel_compatible_df(self) -> pd.DataFrame:
        """
        Specific method for Exporting.
        Creates a copy of the data safe for OpenPyXL writing.
        """
        # copy to avoid modifying the working data
        clean = self.df.copy()

        # 1. Convert Timestamps to Python datetime (OpenPyXL requirement)
        # Select datetime columns efficiently
        for col in clean.select_dtypes(include=['datetime64']).columns:
            clean[col] = np.array(clean[col].dt.to_pydatetime())
            
        # 2. Convert NaN/NaT to None (OpenPyXL requirement for empty cells)
        # We cast to object because None is a Python object
        clean = clean.astype(object).where(pd.notnull(clean), None)
        
        return clean

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