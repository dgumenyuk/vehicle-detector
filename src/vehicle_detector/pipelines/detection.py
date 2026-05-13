from pydantic import BaseModel

from vehicle_detector.helpers import logger
from vehicle_detector.pipelines import BasePipeline


class DetectionPipelineConfig(BaseModel):
    video_path: str
    predictions_dir: str


class DetectionPipeline(BasePipeline):
    def initialize(self, config):
        self.config = config
        logger.info("Initialized detection pipeline.")

    def run(self) -> None:
        logger.info("Running detection pipeline.")
