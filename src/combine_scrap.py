from typing import Dict, Type, Callable, Any, Literal, List, Optional
from abc import ABC, abstractmethod
from importlib.metadata import entry_points
import logging
import json
import pandas as pd
import openpyxl
from enum import Enum
import time
from graphlib import TopologicalSorter, CycleError

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

class ResourceOps(Operation, type_id="resource"):
    
    def __init_subclass__(cls, type_id, **kwargs):
        return super().__init_subclass__(type_id=f"resource.{type_id}", **kwargs)
    
    @classmethod
    def get_subclass(cls, type_id: str) -> Type["ResourceOps"]:
        return cls.get_operation_type(f"resource.{type_id}")

# --- Intermediate Resource Namespaces ---
class FileResource(ResourceOps, type_id="file"): pass
class DatabaseResource(ResourceOps, type_id="db"): pass
class APIResource(ResourceOps, type_id="api"): pass

class Resource(ABC):
    """
    Adapters natively support 'extract' and 'load' based on the JSON op.
    """
    def __init__(self, **kwargs):
        self.cfg = kwargs
        self.client = None
    
    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def disconnect(self): pass

    @abstractmethod
    def read(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def write(self, data: Any, *args, **kwargs) -> None:
        pass

    def __enter__(self):
        self.connect()
    
    def __exit__(self, exc_type, exc, tb):
        self.disconnect()

@FileResource.register_operation("json")
class JsonAdapter(Resource):
    def connect(self):
        file_path = self.cfg.get("path")
        self.client = open(file_path, "r+")
        print(f"Opening Json File: {file_path}")

    def disconnect(self):
        self.client.close()
    
    def read(self, *args, **kwargs):
        content = json.load(self.client)
        return content
    
    def write(self, data, *args, **kwargs):
        # json.dump(self.client, data, indent=4)
        ...

@FileResource.register_operation("excel")
class ExcelAdapter(Resource):
    def connect(self):
        file_path = self.cfg.get("path")
        self.client = openpyxl.load_workbook(file_path)
        print(f"Opening Excel file: {file_path}")
    
    def disconnect(self):
        self.client.close()
    
    def read(self, *args, **kwargs):
        ...
    
    def write(self, data, *args, **kwargs):
        ...

@DatabaseResource.register_operation("postgres")
class PostgresAdapter(Resource):
    def connect(self):
        ...
    
    def disconnect(self):
        ...
    
    def read(self, *args, **kwargs):
        ...
    
    def write(self, data, *args, **kwargs):
        ...

@APIResource.register_operation("rest")
class RestAdapter(Resource):
    def connect(self):
        ...
    
    def disconnect(self):
        ...
    
    def read(self, *args, **kwargs):
        ...
    
    def write(self, data, *args, **kwargs):
        ...

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

OpsFn = Callable[[pd.Series, Dict], pd.Series]

class ElementTransform(TransformOps, type_id="element"):
    pass

@ElementTransform.register_operation("clean_strings")
def clean_strings(series: pd.Series, params: Dict) -> pd.Series:
    """Example function matching the OpsFn signature."""
    to_replace = params.get("replace", "")
    return series.astype(str).str.replace(to_replace, "", regex=False).str.strip()

if __name__ == "__main__":
    fn = ElementTransform.get_operation("clean_strings")
    test_series = pd.Series(["  apple  ", "banana_bad", "cherry"])

    result = fn(test_series, {"replace": "_bad"})
    print("ElementOps Result:")
    print(result.tolist())

Ops = Callable[[pd.DataFrame, Dict], pd.DataFrame]
class SetTransform(TransformOps, type_id="set"):
    pass

@SetTransform.register_operation("filter_rows")
def filter_rows(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """Example function matching the OpsFn signature."""
    col = params.get("column")
    threshold = params.get("threshold", 0)

    if col in df.columns:
        return df[df[col] > threshold]
    return df

if __name__ == "__main__":
    fn = SetTransform.get_operation("filter_rows")
    test_df = pd.DataFrame({"A": [1,5,10], "B": [2,2,2]})

    result = fn(test_df, {"column": "A", "threshold": 4})
    print("SetTransform Result:\n", result)

OpsFn = Callable[[List[pd.DataFrame], Dict], pd.DataFrame]
class CollectionTransform(TransformOps, type_id="collection"):
    pass

@CollectionTransform.register_operation("test_fn")
def test_fn(dfs: List[pd.DataFrame], params: Dict) -> pd.DataFrame:
    test_data = {
        "A": [1, 4, 10],
        "B": [2, 2, 2]
    }
    return pd.DataFrame(test_data)

if __name__ == "__main__":
    fn = CollectionTransform.get_operation("test_fn")
    result = fn([], {})
    print("CollectionTransform Result:\n", result)

class TransferOps(Operation, type_id="transfer"):
    """Namespace for generic transfer actions (extract, load)."""

    _CONNECTION_REGISTRY: Dict[str, Resource] = {}

    @classmethod
    def register_resource(cls, id: str, resource: Resource):
        if id:
            if id in cls._CONNECTION_REGISTRY:
                raise ValueError(f"Connection '{id}' already established.")
            cls._CONNECTION_REGISTRY[id] = resource
    
    @classmethod
    def get_resource(cls, id: str) -> Resource:
        if id not in cls._CONNECTION_REGISTRY:
            raise KeyError(f"Unknown resource connection: {id}")
        return cls._CONNECTION_REGISTRY[id]
    
    @classmethod
    def clear_connections(cls):
        """
        Safely closes and removes all connections.
        Crucial for avoiding global state leaks across different pipeline runs.
        """
        for res_id, conn in cls._CONNECTION_REGISTRY.items():
            conn.disconnect()
        cls._CONNECTION_REGISTRY.clear()

# --- Registered Operations ---

@TransferOps.register_operation("extract")
def _extract(resource: str, *args, **kwargs) -> Dict[str, Any]:
    adapter = TransferOps.get_resource(resource)
    with adapter:
        data = adapter.read(*args, **kwargs)
        data = pd.json_normalize(data)
    return pd.DataFrame(data)

@TransferOps.register_operation("load")
def _load(resource: str, data: pd.DataFrame, *args, **kwargs) -> None:
    adapter = TransferOps.get_resource(resource)
    with adapter:
        adapter.write(data=data.to_dict(), *args, **kwargs)

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class Task:
    def __init__(
        self,
        id: str,
        action: Callable,
        params: Dict[str, Any] = {},
        attempts: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        metadata: Optional[Dict] = None
    ):
        self.id = id
        self.action = action
        self.params = params
        self.attempts = attempts
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor

        # State & Telemetry
        self.status = TaskStatus.PENDING
        self.output: Any = None
        self.error: Optional[Exception] = None
        self.metadata = metadata or {}

        # Metrics
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_time: float = 0.0
    
    def run(self, *args, **kwargs) -> Any:
        """Execution Wrapper handling task lifecycle with exponential backoff."""
        self.status = TaskStatus.RUNNING
        self.start_time = time.time()
        log.info(f"Task '{self.id}' started.")

        remaining_attempts = self.attempts
        current_delay = self.retry_delay

        while remaining_attempts > 0:
            try:
                merged_args = dict(**kwargs, **self.params)
                print(merged_args, args)
                # Execute action logic
                self.output = self.action(*args, **merged_args)
                self.status = TaskStatus.COMPLETED
                break

            except Exception as e:
                remaining_attempts -= 1

                if remaining_attempts == 0:
                    self.status = TaskStatus.FAILED
                    self.error = e
                    self._finalize_metrics()
                    log.error(f"Task '{self.id}' failed permanently. Error: {e}", exc_info=True)
                    raise e
                else:
                    log.warning(f"Task '{self.id}' failed. Retrying in {current_delay}s. Attempts left: {remaining_attempts}. Error: {e}")
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor    # Exponential backoff
        self._finalize_metrics()
        log.info(f"Task '{self.id}' finished in {self.elapsed_time} seconds with status: {self.status.value}")
        return self.output
    
    def _finalize_metrics(self):
        self.end_time = time.time()
        if self.start_time:
            self.elapsed_time = round(self.end_time - self.start_time, 4)
    
    def __repr__(self):
        return f"<Task id={self.id} status={self.status.value}>"

class Pipeline:
    def __init__(self, id: str):
        self.id = id
        self.tasks: Dict[str, Task] = {}
        self._task_dag: Dict[str, List[str]] = {}
    
    def add_task(self, task: Task, depends_on: Optional[List[str]] = None):
        """Registers a task and its upstream dependencies."""
        self.tasks[task.id] = task
        self._task_dag[task.id] = list(set(depends_on)) if depends_on else []
    
    def run(self):
        """Executes tasks in the correct topological order."""
        log.info(f"Starting Pipeline: {self.id}")

        # Validation 1: Ensure all dependencies are registered as tasks
        all_deps = {dep for deps in self._task_dag.values() for dep in deps}
        missing_deps = all_deps - set(self.tasks.keys())
        if missing_deps:
            raise ValueError(f"Tasks {missing_deps} are dependencies but were never added to the pipeline.")
        
        ts = TopologicalSorter(self._task_dag)

        # Validation 2: Explicitly check for cyclic dependencies before running
        try:
            execution_order = tuple(ts.static_order())
        except CycleError as e:
            log.critical(f"Pipeline validation failed: Cyclic dependency detected. {e}")
            raise

        # Execution Phase
        for task_id in execution_order:
            try:
                self._execute_task(task_id)
            except Exception as e:
                log.critical(f"Pipeline execution halted at task '{task_id}' due to: {e}")
        
        log.info(f"Pipeline '{self.id}' completed successfully.")
    
    def _execute_task(self, id: str):
        task = self.tasks[id]
        deps = self._task_dag[id]

        # Collect outputs from parent tasks
        deps_output = [self.tasks[dep].output for dep in deps]

        if deps_output:
            task.run(*deps_output)
        else:
            task.run()