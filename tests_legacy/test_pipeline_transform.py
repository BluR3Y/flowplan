from batch_etl.transforms import TransformTask, ElementTransform
import pandas as pd

def test_transform_registry():
    # print(TransformOperation._LEVEL_REGISTRY)
    # level_ops = TransformOperation.get_level_ops("element")
    # print(level_ops._ELEMENT_OPERATION_REGISTRY)
    my_input = pd.DataFrame({
        "grant_id": ["jjc-100", "jjc-101", "jjc-102", "jjc-103"],
        "status": ["Open", "Closed", "Closed", "Open"],
        "title": ["ipsum lorem", "lorem Ipsum", "Ipsum Lorem", "Lorem ipsum"]
    })
    my_task = {
        "task_id": "tester_1",
        "level": "element",
        "input": ["excel_1_sheet_1"],
        "steps": [
            { "op": "cast", "on_error": "fail", "target": { "grant_id": "integer" } }
        ]
    }
    # print(TransformOperation._LEVEL_REGISTRY)
    # transform_ops = TransformOperation.get_level_ops(my_task.get("level"))
    # print(transform_ops)
    transform_ops = TransformTask.get_level_ops(my_task.get("level"))
    print(transform_ops)