from abc import ABC, abstractmethod
from importlib.metadata import entry_points
from typing import Dict, Any, Literal, Callable, List, Type
import logging

log = logging.getLogger(__name__)

class PipelineTask(ABC):
    _STATUS: Literal["PENDING", "SUCCESS", "FAILED"] = None
    _TASK_REGISTRY: Dict[str, Type["PipelineTask"]] = {}

    def __init__(self, task_id: str, **kwargs):
        self.id = task_id
        self._STATUS = "PENDING"
        self.cfg = kwargs

    def __init_subclass__(cls, type_id: str):
        if type_id is not None:
            if type_id in cls._TASK_REGISTRY:
                raise ValueError(f"Task Type '{type_id}' already registered.")
            cls._TASK_REGISTRY[type_id] = cls
    
    @classmethod
    def get_task_subclass(cls, type_id: str):
        if type_id not in cls._TASK_REGISTRY:
            raise ValueError(f"Unknown task type: {type_id}")
        return cls._TASK_REGISTRY[type_id]

    @abstractmethod
    def run_task(self):
        ...
    
    @classmethod
    def _load_plugins(cls, group_name: str, registry: Dict[str, Callable], on_exist: Literal["skip", "overwrite"] = "skip"):
        """Load external plugins registered via entry points."""
        try:
            eps = entry_points()
            full_name = '.'.join(["batch_etl", group_name])

            # Support Python 3.10+ select() and older dict interface
            candidates = eps.select(group=full_name) if hasattr(eps, "select") else eps.get(full_name, [])
            
            for ep in candidates:
                if ep.name not in registry or on_exist == "overwrite":
                    try:
                        # Load plugin (executes module level code/decorators)
                        plugin_fn = ep.load()

                        # If the plugin didn't use the decorator but just returned a callable.
                        # we register it manually here
                        if callable(plugin_fn):
                            registry[ep.name] = plugin_fn
                        log.debug(f"Loaded '{group_name}' plugin '{ep.name}'")
                    except Exception as e:
                        log.warning(f"Failed to load '{group_name}' plugin '{ep.name}': {e}")
                else:
                    log.warning(f"Skipped '{group_name}' plugin '{ep.name}' import")
        except Exception as e:
            log.debug(f"Plugin loading failed: {e}")