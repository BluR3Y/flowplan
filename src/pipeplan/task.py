import logging
import time
from typing import Callable, Optional, Dict, Any
from enum import Enum

log = logging.getLogger(__name__)

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