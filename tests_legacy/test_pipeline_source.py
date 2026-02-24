# from batch_etl.sources.base_adapter import SourceAdapter
# from batch_etl.sources.json_adapter import JsonAdapter
# from batch_etl.sources.access_adapter import AccessAdapter
# from batch_etl.sources.excel_adapter import ExcelAdapter

# from batch_etl.sources import SourceAdapter, JsonAdapter, AccessAdapter, ExcelAdapter
from batch_etl.extract import ExtractTask

# def test_source_registry():
#     test_config = {
#         "path": "C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json",
#         "data": {
#             "activity_types": {
#                 "dataset_id": "activity",
#                 "columns": ["Name","Code"]
#             }
#         }
#     }
#     adapter = SourceAdapter.extract(test_config)
#     assert isinstance(adapter, JsonAdapter)
#     print(adapter.load_data())

def test_source_registry():
    test_extract = {
        "task_id": "json_extract_test",
        "connect": {
            "adapter": "jsson",
            "params": {
                "path": "C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json"
            }
        }
    }
    my_extract = ExtractTask(test_extract)
    my_extract.run_task()