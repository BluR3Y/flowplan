from pipeplan import Pipeline, Task, Operation

def test_pipeline_orchestration():
    res_type = Operation.get_operation_type("resource")
    res_sub = res_type.get_subclass("file")
    res_cls = res_sub.get_operation("json")
    res_obj = res_cls(path="C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json")

    transfer_type = Operation.get_operation_type("transfer")
    transfer_type.register_resource("test_conn_1", res_obj)
    transfer_op = transfer_type.get_operation("extract")
    
    test_task = Task(
        id="test_task_1",
        action=transfer_op,
        params={
            "resource":"test_conn_1"
        }
    )
    
    test_pl = Pipeline("test_pl_1")
    test_pl.add_task(test_task)
    test_pl.run()
    print(test_pl.tasks["test_task_1"].output)