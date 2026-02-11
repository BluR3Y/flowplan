from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable

class TransformOperation:
    _LEVEL_REGISTRY: Dict[str, Type["TransformOperation"]] = {}

    # Missing init

    def __init_subclass__(cls, type_id, **kwargs):
        super().__init_subclass__(**kwargs)
        if type_id is not None:
            if type_id in cls._LEVEL_REGISTRY:
                raise ValueError(f"Transform operation type '{type_id}' already registered.")
            cls._LEVEL_REGISTRY[type_id] = cls

    @classmethod
    def get_level_ops(cls, level: str):
        """Retrieve transform operation by level"""
        if level not in cls._LEVEL_REGISTRY:
            raise ValueError(f"Unknown transform level: {level}")
        
        return cls._LEVEL_REGISTRY[level]
    
    @abstractmethod
    def register_operation(name: str):
        ...

    @abstractmethod
    def get_operation(name: str):
        ...


if __name__ == "__main__":
    my_task = {
        "level": "element",
        "input": ["excel_1_sheet_1"],
        "steps": [
            { "op": "cast", "on_error": "fail", "target": { "grant_id": "integer" } }
        ]
    }
    print(TransformOperation._LEVEL_REGISTRY)
    transform_ops = TransformOperation.get_level_ops(my_task.get("level"))
    print(transform_ops)