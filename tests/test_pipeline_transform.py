from batch_etl.transforms.base_level import TransformOperation

def test_transform_registry():
    level_ops = TransformOperation.get_level_ops("element")
    print(level_ops)