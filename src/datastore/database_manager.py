from typing import Type

from .collection_manager import CollectionManager
from .table_manager import TableManager

class DatabaseManager(CollectionManager):
    _document_class: Type[TableManager] = TableManager

    def __init__(self, read_file_path: str) -> None:
        if not read_file_path:
            raise ValueError("Database file path was not provided.")

        super().__init__()
        self.read_file_path = read_file_path
        