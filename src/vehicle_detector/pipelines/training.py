from fontTools import log
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel
from ultralytics import YOLO

from vehicle_detector.helpers import logger
from vehicle_detector.helpers.training_helpers import train_test_split
from vehicle_detector.pipelines import BasePipeline


class TrainingPipelineConfig(BaseModel):
    train_dataset_path: str
    model_save_path: str
    train_percentage: float = 0.8
    class_name: str = "vehicle"
    yolo_model_config: str = "yolo26s.yaml"


class TrainingPipeline(BasePipeline):
    def initialize(self, config):
        self.config = TrainingPipelineConfig(**config)
        logger.info("Initialized training pipeline.")

    def run(self) -> None:
        logger.info("Running training pipeline.")
        # logger.info("Starting train-test split.")
        # self.train_test_split()
        # logger.info("Train-test split completed.")
        logger.info("Starting model training.")
        self.train_model()

    def train_test_split(self) -> None:
        """Splits the dataset into a training and a test set."""
        dataset_path = Path(self.config.train_dataset_path)

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        save_dir = dataset_path / "train_val_splits"
        train_test_split(
            images_dir,
            labels_dir,
            save_dir,
            train_percentage=self.config.train_percentage,
            class_name=self.config.class_name,
        )

    def train_model(self) -> None:
        """Trains the model using the training data."""
        dataset_yaml_path = Path(self.config.train_dataset_path) / "train_val_splits" / "dataset.yaml"
        model = YOLO(self.config.yolo_model_config)
        model_id: str =  datetime.now().strftime("%Y%m%d_%H%M%S")
        model.train(data=dataset_yaml_path, epochs=50, project=self.config.model_save_path, name=model_id, mosaic=0, pretrained=False, translate=0, scale=0.1)
