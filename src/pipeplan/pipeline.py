import logging
from typing import List, Dict, Optional
from graphlib import TopologicalSorter, CycleError
from .task import Task

log = logging.getLogger(__name__)

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

# --- Example Usage ---
if __name__ == "__main__":
    pl = Pipeline("data_processing_pipeline")
    
    def extract_data():
        return {"users": ["Alice", "Bob"]}
        
    def transform_data(data):
        # Intentional error simulated to show retry logic
        import random
        if random.random() < 0.5:
             raise ConnectionError("Simulated network blip!")
        data["users"] = [u.upper() for u in data["users"]]
        return data

    def load_data(data):
        print(f"Loading data to DB: {data}")

    # Create tasks
    t1 = Task("extract", action=extract_data)
    t2 = Task("transform", action=transform_data, attempts=3, retry_delay=1)
    t3 = Task("load", action=load_data)

    # Add to pipeline with dependencies
    pl.add_task(t1)
    pl.add_task(t2, depends_on=["extract"])
    pl.add_task(t3, depends_on=["transform"])

    # Run pipeline
    pl.run()