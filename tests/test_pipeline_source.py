# from batch_etl.sources.base_adapter import SourceAdapter
# from batch_etl.sources.json_adapter import JsonAdapter
# from batch_etl.sources.access_adapter import AccessAdapter
# from batch_etl.sources.excel_adapter import ExcelAdapter

from batch_etl.sources import SourceAdapter, JsonAdapter, AccessAdapter, ExcelAdapter

def test_source_registry():
    test_config = {
        "path": "C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json",
        "data": {
            "activity_types": {
                "dataset_id": "activity",
                "columns": ["Name","Code"]
            }
        }
    }
    adapter = SourceAdapter.extract(test_config)
    assert isinstance(adapter, JsonAdapter)
    print(adapter.load_data())