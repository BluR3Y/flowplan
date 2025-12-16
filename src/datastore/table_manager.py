
from .document_manager import DocumentManager

class TableManager(DocumentManager):
    def __init__(self, db_path: str):
        super().__init__()