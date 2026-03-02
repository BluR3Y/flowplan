from abc import ABC, abstractmethod
from typing import Dict, Literal, Any

# class Resource(ABC):
    
#     def __init__(self, **kwargs):
#         self.cfg = kwargs
    
#     @abstractmethod
#     def _extract(self, **kwargs) -> Dict[str, Any]:
#         ...
    
#     @abstractmethod
#     def _load(self, **kwargs) -> None:
#         ...
    
#     def __call__(self, action: Literal["extract", "load"], **kwargs):
#         if action == "extract":
#             return self._extract(**kwargs)
#         elif action == "load":
#             return self._load(**kwargs)
#         else:
#             raise ValueError(f"Unknown resource action: {action}")

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