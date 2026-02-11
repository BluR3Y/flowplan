from __future__ import annotations
from typing import Dict, Optional, Sequence, List, Type, Set

import logging
import pandas as pd

from .document_manager import DocumentManager
log = logging.getLogger(__name__)

class CollectionManager:
    """
    Generic manager for a collection of names DataFrames (Documents).
    """
    # Allow subclasses to specify what type of manager they create
    _document_class: Type[DocumentManager] = DocumentManager

    def __init__(self) -> None:
        self._docs_created: Dict[str, DocumentManager] = {}
        self._docs_deleted: Set[str] = set()
        self._dirty: bool = False
    
    def mark_dirty(self) -> None:
        self._dirty = True

    def create_document(self, name: str, data: DocumentManager.DataType) -> DocumentManager:
        """
        Registers a new dataframe. Uses the class defined in _document_class.
        """
        if name in self._docs_deleted:
            self._docs_deleted.remove(name)

        # We use self._document_class so subclasses can automatically create correct type of manager.
        doc = self._document_class(name, data, on_change=self.mark_dirty)
        self._docs_created[name] = doc
        self.mark_dirty()
        return doc
    
    def delete_document(self, name: str) -> DocumentManager:
        if name not in self._docs_created:
            raise KeyError(f"Document '{name}' not found in collection.")
        
        self._docs_deleted.add(name)
        self.mark_dirty()
        return self._docs_created.pop(name)
    
    def get_document(self, name: str) -> DocumentManager:
        if name not in self._docs_created:
            raise KeyError(f"Document '{name}' not found in collection.")
        return self._docs_created[name]
    
    def list_documents(self) -> list[str]:
        return list(self._docs_created.keys())