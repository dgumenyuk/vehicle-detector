from pydantic import BaseModel

from vehicle_detector.helpers import logger
from vehicle_detector.pipelines import BasePipeline


class EvaluationPipelineConfig(BaseModel):
    test_data_dir: str


class EvaluationPipeline(BasePipeline):
    def initialize(self, config):
        logger.info("Initialized evaluation pipeline.")
        self.config = config

    def run(self) -> None:
        pass
