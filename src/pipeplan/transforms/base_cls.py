from ..operation import Operation
from typing import Type

class TransformOps(Operation, type_id="transform"):
    """
    Intermediate base class for all transformations.
    Automatically prepends 'transform.' to all child operation types.
    """

    def __init_subclass__(cls, type_id: str, **kwargs):
        return super().__init_subclass__(type_id=f"transform.{type_id}", **kwargs)
    
    @classmethod
    def get_subclass(cls, type_id: str) -> Type["TransformOps"]:
        """Helper to retrieve children using only their short name."""
        return cls.get_operation_type(f"transform.{type_id}")