from ..pipeline_task import PipelineTask
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Callable

class TransformTask(PipelineTask, type_id="transform"):
    _LEVEL_REGISTRY: Dict[str, Type["TransformTask"]] = {}

    def __init__(self, task_id, **kwargs):
        super().__init__(task_id, **kwargs)

    def __init_subclass__(cls, type_id: str, **kwargs):
        super().__init_subclass__(**kwargs)
        if type_id is not None:
            if type_id in cls._LEVEL_REGISTRY:
                raise ValueError(f"Transform operation type '{type_id}' already registered.")
            cls._LEVEL_REGISTRY[type_id] = cls

    @classmethod
    def get_transform(cls, level: str):
        """Retrieve transform instance by granularity"""
        if level not in cls._LEVEL_REGISTRY:
            raise ValueError(f"Unknown transform level: {level}")
        return cls._LEVEL_REGISTRY[level]
    
    @classmethod
    @abstractmethod
    def register_operation(cls, name: str):
        ...

    @classmethod
    @abstractmethod
    def get_operation(self, name: str):
        ...

    # @classmethod
    # @abstractmethod
    # def apply_transform_steps(self, )