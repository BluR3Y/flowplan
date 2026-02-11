import logging
import sys
from abc import ABC, abstractmethod
from typing import List

logger = logging.getLogger(__name__)

class TaskDAG:
    def __init__(self):
        self.task_id = None
        self.data = {}
        self.next = None