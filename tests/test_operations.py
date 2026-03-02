import pipeplan
from pipeplan.operation import Operation
from pipeplan.transforms import TransformOps

def test_operation_registry():
    ops_type = Operation.get_operation_type("transform")
    ops_sub = ops_type.get_subclass("element")
    ops_fn = ops_sub.get_operation("clean_strings")
    print(ops_fn)