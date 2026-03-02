from typing import Dict, Type, Callable, Any
from abc import ABC
from importlib.metadata import entry_points
import logging

log = logging.getLogger(__name__)

class Operation(ABC):
    """
    The master registry for all pipeline operations.
    Handles subclass registration and lazy-loads plugins dynamically.
    """

    # Registry for the Operation Subclasses
    _OPERATION_REGISTRY: Dict[str, Type["Operation"]] = {}

    def __init_subclass__(cls, type_id: str, **kwargs):
        """Automatically registers subclasses and sets up their internal registries."""
        super().__init_subclass__(**kwargs)
        
        if type_id:
            if type_id in cls._OPERATION_REGISTRY:
                raise ValueError(f"Operation type '{type_id}' already registered.")
            
            cls._OPERATION_REGISTRY[type_id] = cls

            # Dynamically create a specific function registry for this subclass
            cls._function_registry: Dict[str, Callable] = {}

            cls._plugins_loaded = False

    @classmethod
    def _load_plugins(cls, type_id: str):
        """Lazy loads external plugins safely."""
        if getattr(cls, "_plugins_loaded", False):
            return
        
        try:
            group_name = '.'.join(["pipeplan", type_id])
            eps = entry_points()
            candidates = eps.select(group=group_name) if hasattr(eps, "select") else eps.get(group_name, [])

            for ep in candidates:
                try:
                    plugin_fn = ep.load()

                    if callable(plugin_fn) and ep.name not in cls._function_registry:
                        cls.register_operation(ep.name)(plugin_fn)
                    
                    log.debug(f"Loaded '{type_id}' plugin '{ep.name}'")
                except Exception as e:
                    log.warning(f"Failed to load '{group_name}' plugin '{ep.name}': {e}")
        except Exception as e:
            log.warning(f"Plugin import failed: {e}")
        finally:
            cls._plugins_loaded = True
    
    @classmethod
    def register_operation(cls, name: str):
        """Decorator to register a function to this specific Operation type."""
        def decorator(fn: Callable):
            if name in cls._function_registry:
                raise ValueError(f"Operation '{name}' already exists in {cls.__name__}")
            
            cls._function_registry[name] = fn
            return fn
        return decorator
    
    @classmethod
    def get_operation(cls, name: str) -> Callable:
        """Retrieve a registered function."""
        # Lazy Load Trigger: Only load plugins the first time an operation is requested
        if not getattr(cls, "_plugins_loaded", False):
            # Find the type_id we were registered under
            for t_id, t_cls in cls._OPERATION_REGISTRY.items():
                if t_cls is cls:
                    cls._load_plugins(t_id)
                    break

        if name not in cls._function_registry:
            raise KeyError(f"Unknown operation '{name}' in {cls.__name__}")
        return cls._function_registry[name]

    @classmethod
    def get_operation_type(cls, type_id: str) -> Type["Operation"]:
        """Retrieve a registered subclass by its type_id."""
        if type_id not in cls._OPERATION_REGISTRY:
            raise KeyError(f"Unknown operation type: {type_id}")
        return cls._OPERATION_REGISTRY[type_id]