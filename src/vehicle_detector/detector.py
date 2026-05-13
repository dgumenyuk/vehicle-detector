import yaml
from pydantic import BaseModel

from vehicle_detector.helpers import logger
from vehicle_detector.pipelines import (
    BasePipeline,
    DetectionPipeline,
    DetectionPipelineConfig,
    EvaluationPipeline,
    EvaluationPipelineConfig,
    LabelingPipeline,
    LabelingPipelineConfig,
    TrainingPipeline,
    TrainingPipelineConfig,
)


class VehicleDetectorConfig(BaseModel):
    training: TrainingPipelineConfig | None = None
    detection: DetectionPipelineConfig | None = None
    evaluation: EvaluationPipelineConfig | None = None
    labeling: LabelingPipelineConfig | None = None


class VehicleDetector:
    """Main class for the vehicle detector."""

    def __init__(self, config):
        self.config_path = config
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

        self.config = VehicleDetectorConfig(**self.config)
        self.pipelines: dict[str, BasePipeline] = {}
        for pipeline_name, pipeline_config in self.config.model_dump().items():
            if pipeline_config is not None:
                pipeline_instance: BasePipeline = self.get_pipeline_class(pipeline_name)
                pipeline_instance.initialize(pipeline_config)
                setattr(self, f"{pipeline_name}_pipeline", pipeline_instance)
                self.pipelines[pipeline_name] = pipeline_instance

    def get_pipeline_class(self, pipeline_name: str) -> BasePipeline:
        """Get the pipeline class for the given pipeline name."""
        pipeline_classes: dict[str, BasePipeline] = {
            "training": TrainingPipeline(),
            "detection": DetectionPipeline(),
            "evaluation": EvaluationPipeline(),
            "labeling": LabelingPipeline(),
        }
        return pipeline_classes[pipeline_name]

    def run(self) -> None:
        """Detect vehicles in the given image and return a list of bounding boxes."""
        for pipeline_names, pipeline in self.pipelines.items():
            logger.info(f"Running {pipeline_names} pipeline")
            pipeline.run()
