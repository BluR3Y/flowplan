from ..adapter_registry import FileResource
from ..base_adapter import Resource
import openpyxl

@FileResource.register_operation("excel")
class ExcelAdapter(Resource):
    def connect(self):
        file_path = self.cfg.get("path")
        self.client = openpyxl.load_workbook(file_path)
        print(f"Opening Excel file: {file_path}")
    
    def disconnect(self):
        self.client.close()
    
    def read(self, *args, **kwargs):
        ...
    
    def write(self, data, *args, **kwargs):
        ...