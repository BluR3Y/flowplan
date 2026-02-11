from abc import ABC, abstractmethod

class ETLTask(ABC):
    def __init__(self):
        self.id = None
        self.data = None

    @abstractmethod
    def run(self):
        ...