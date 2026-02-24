from abc import ABC
from typing import Dict, Any, List, Set, Type
from graphlib import TopologicalSorter 
from .pipeline_task import PipelineTask

# Pipeline Orchestrator using DAGs
class PipelineManager(ABC):
    _TASK_REGISTRY: Dict[str, Type[PipelineTask]] = None
    _TASK_DAG: Dict[str, Set[str]] = None
    
    def __init__(self, name: str):
        self.id = name
        self._TASK_DAG = {}
        self._TASK_REGISTRY = {}

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def add_task(self, task: PipelineTask, input: List[str] = None):
        self._TASK_REGISTRY[task.id] = task
        self._TASK_DAG[task.id] = set()
        if input:
            for dep_id in input:
                self._TASK_DAG[task.id].add(dep_id)
    
    def run(self):
        # Topological sort ensures we follow dependencies
        ts = TopologicalSorter(self._TASK_DAG)
        for task_id in tuple(ts.static_order()):
            self._execute_task(self._TASK_REGISTRY[task_id])
    
    def _execute_task(self, task: PipelineTask):
        print(f"Executing {task.id}...")
        try:
            task.run_task()
            task.status = "SUCCESS"
        except Exception as e:
            # Production requirement: Implement Retry Logic here
            task.status = "FAILED"
            print(f"Tasl {task.id} failed: {e}")