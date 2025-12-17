import pyodbc

from pathlib import Path

from typing import Type, Optional, Sequence, Union

from .collection_manager import CollectionManager
from .table_manager import TableManager

class DatabaseManager(CollectionManager):
    # Tell the parent class to create TableManagers, not DocumentManagers
    _document_class = TableManager

    def __init__(
        self,
        read_file_path: str,
        *,
        include_tables: Optional[Sequence[str]] = None
    ) -> None:
        if not read_file_path:
            raise ValueError("Database file path was not provided.")

        # super().__init__()
        # self.read_file_path = read_file_path
        # self.include_tables = include_tables
        path_obj = Path(read_file_path)
        if not path_obj.exists():
            raise ValueError(f"No database exists at path: {read_file_path}")
        if not path_obj.is_file():
            raise TypeError("Path does not point to a file.")
        
        super().__init__()
        self.db_path_obj = path_obj
        self.include    # Last Here: setting up init func
        self.db_driver = self._get_db_driver(path_obj.suffix)
        self.connection = None
        self.cursor = None
    
    @staticmethod
    def _get_db_driver(db_ext) -> str:
        driver_mapping = {
            (".mdb", ".accdb"): "Microsoft Access Driver (*.mdb, *.accdb)",
            (".db", ".sqlite"): "SQLite3 ODBC Driver"
        }
        db_driver = next(driver for extensions, driver in driver_mapping.items() if db_ext in extensions)
        if not db_driver:
            raise TypeError("Database not supported.")
        return db_driver

    def init_db_connection(self):
        """Initialize database connection."""
        self.connection = pyodbc.connect()