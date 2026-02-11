from .base_task import ETLTask
import logging

logger = logging.getLogger(__name__)

class TransformTask(ETLTask):
    def __init__(self, data):
        self.data = data
    
    def run(self):
        logger.info("Transforming data")
        return