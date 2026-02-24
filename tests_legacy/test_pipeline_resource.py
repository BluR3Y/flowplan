from batch_etl.resources import PipelineResource


    # conn_adapter = PipelineResource.get_resource_type(my_resource.get("adapter"))
    # conn_obj = conn_adapter(**my_resource.get("params", {}))
    # conn_obj.extract_data()

def test_pipeline_resource():
    my_resources = [
        {
            "connection_id": "test_input_conn",
            "adapter": "json",
            "params": {
                "path": "C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_data.json"
            }
        },
        {
            "connection_id": "test_output_conn",
            "adapter": "json",
            "params": {
                "path": "C:/Users/reyhe/OneDrive/Documents/GitHub/flowplan/dev_docs/blueprints/misc_output_data.json"
            }
        }
    ]
    # conn_adapter = PipelineResource.get_resource_type(my_resource.get("adapter"))
    # conn_obj = conn_adapter(my_resource.get("params", {}))
    # print(conn_obj.extract_data(records="activity_tyspes"))
    my_connections: dict[str, PipelineResource] = {}
    for conn in my_resources:
        conn_adapter = PipelineResource.get_resource_type(conn.get("adapter"))
        conn_obj = conn_adapter(conn.get("params", {}))
        my_connections[conn.get("connection_id")] = conn_obj
    
    input_conn = my_connections.get("test_input_conn")
    input_data = input_conn.extract_data(records="activity_types")
    output_conn = my_connections.get("test_output_conn")
    output_conn.load_data(input_data, records="activity_types")