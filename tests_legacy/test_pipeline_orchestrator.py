from batch_etl.pipeline_manager import PipelineManager
from batch_etl.pipeline_task import PipelineTask

def test_pipeline_registry():
    test_config = {
        "connections": [    # Credentials
            {
                "connection_id": "excel_conn_1",
                "resource": "doc",  # api | db | doc
                "adapter": "excel", # resource is a 'collection'
                "params": {
                    "path": "D:/Data/excel_data.xlsx"
                }
            },
            {
                "connection_id": "json_conn_1",
                "resource": "doc",
                "adapter": "json",  # resource is a 'set'
                "params": {
                    "path": "D:/Data/json_data.json"
                }
            }
        ],
        "tasks": [
            {
                "task_id": "test_excel_extract_1",
                "phase": "extract",
                "resource": ["excel_conn_1"],
                "configure": {
                    "collection": "grants_sheet",
                    # "keys": [
                    #     "grant_id",
                    #     "pln_id"
                    # ],
                    "params": {}
                    # "condition": {
                    #     "AND": [
                    #         {
                    #             "status": {
                    #                 "op": "==",
                    #                 "value": "Active"
                    #             }
                    #         },
                    #         {
                    #             "award_number": {
                    #                 "op": "!=",
                    #                 "value": None
                    #             }
                    #         }
                    #     ]
                    # }
                }
            },
            {
                "task_id": "test_json_extract_1",
                "phase": "extract",
                "resource": ["json_conn_1"],
                "configure": {
                    "keys": ["Name", "Code"]
                }
            },
            {
                "task_id": "test_elem_trans_1",
                "phase": "transform",
                "input": ["test_excel_extract_1"],
                "configure": {
                    "level": "element",
                    "steps": []
                }
            },
            {
                "task_id": "test_json_load_1",
                "phase": "load",
                "input": ["test_elem_trans_1"],
                "resource": ["json_conn_1"],
                "configure": {
                    "mode": "overwrite"
                }
            }
        ]
    }
    with PipelineManager("test_pipeline") as pm:
        for task in test_config.get("tasks", []):
            task_cls = PipelineTask.get_task_subclass(task.get("phase"))
            task_obj = task_cls(task.get("task_id"))
            pm.add_task(task_obj)