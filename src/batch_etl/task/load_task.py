from .base_task import ETLTask
import logging

logger = logging.getLogger(__name__)

class LoadTask(ETLTask):
    def __init__(self, data):
        self.data = data
    
    def run(self):
        logger.info("Loading dataset")
        return