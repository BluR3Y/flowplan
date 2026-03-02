from ...resources import FileResource
from ...resources.base_adapter import Resource
import json

@FileResource.register_operation("json")
class JsonAdapter(Resource):
    def connect(self):
        file_path = self.cfg.get("path")
        self.client = open(file_path, "r+")
        print(f"Opening Json File: {file_path}")

    def disconnect(self):
        self.client.close()
    
    def read(self, *args, **kwargs):
        content = json.load(self.client)
        return content
    
    def write(self, data, *args, **kwargs):
        # json.dump(self.client, data, indent=4)
        ...