from pathlib import Path

from vehicle_detector.detector import VehicleDetector
from vehicle_detector.pipelines import (
    EvaluationPipeline,
    LabelingPipeline,
    TrainingPipeline,
)

if __name__ == "__main__":
    config_path: Path = Path("example") / "pipeline_config.yaml"
    vehicle_detector = VehicleDetector(config_path)
    labeling_pipeline: LabelingPipeline = vehicle_detector.labeling_pipeline
    #labeling_pipeline.run()
    #training_pipeline: TrainingPipeline = vehicle_detector.training_pipeline
    # training_pipeline.run()
    evaluation_pipeline: EvaluationPipeline = vehicle_detector.evaluation_pipeline
    #evaluation_pipeline.run()
    evaluation_pipeline.visualize_predictions()
    # labeling_pipeline.visualize_predictions()
    # labeling_pipeline.run()
    # labeling_pipeline.remove_items_by_id(item_id_list=["train_C_640_360_24fps_000004", "train_C_640_360_24fps_000005", "train_C_640_360_24fps_000006",
    #                   "train_C_640_360_24fps_000007", "train_C_640_360_24fps_000008", "train_C_640_360_24fps_000009",
    #                   "train_C_640_360_24fps_000010", "train_C_640_360_24fps_000012", "train_C_640_360_24fps_000013",
    #                   "train_C_640_360_24fps_000015"])
