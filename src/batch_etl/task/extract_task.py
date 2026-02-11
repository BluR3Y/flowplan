from .base_task import ETLTask
import logging

logger = logging.getLogger(__name__)

class ExtractTask(ETLTask):
    def run(self):
        logger.info("Extracting data from source")
        return