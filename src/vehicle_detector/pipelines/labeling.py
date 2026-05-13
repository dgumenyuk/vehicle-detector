import os
from pathlib import Path

from pydantic import BaseModel

from vehicle_detector.helpers import logger
from vehicle_detector.helpers.labeling_helpers import (
    extract_frames,
    label_frames,
    remove_images_without_objects,
    remove_items_by_id,
    run_yolo_on_video,
    visualize_lables,
)
from vehicle_detector.pipelines import BasePipeline


class LabelingPipelineConfig(BaseModel):
    train_video_folder_path: str
    train_dataset_path: str
    eval_video_folder_path: str
    eval_dataset_path: str
    yolo_model_path: str | None = None
    conf_threshold: float = 0.25
    extraction_framerate: int = 5
    max_sizes: dict | None = None
    min_sizes: dict | None = None


class LabelingPipeline(BasePipeline):
    def initialize(self, config: dict) -> None:
        """Initialization function.

        Args:
            config (dict): Labeling pipeline configuration dictionary.
        """
        self.config = LabelingPipelineConfig(**config)
        logger.info("Initialized LabelingPipeline.")

    def run(self) -> None:
        logger.info("Running labeling pipeline")

        data_sources = [
            (
                "train",
                self.config.train_video_folder_path,
                self.config.train_dataset_path,
            ),
            ("eval", self.config.eval_video_folder_path, self.config.eval_dataset_path),
        ]

        for split_name, video_folder, dataset_path in data_sources:
            self._process_dataset_split(split_name, video_folder, dataset_path)

    def visualize_predictions(self, eval: bool = False):
        """This functions visualized the predicted boxes and their sizes with opencv."""
        if eval:
            for video in os.listdir(self.config.eval_video_folder_path):
                video_path = os.path.join(self.config.eval_video_folder_path, video)
                run_yolo_on_video(
                    video_path=video_path,
                    model_path=self.config.yolo_model_path,
                    conf=self.config.conf_threshold,
                )
        else:
            for video in os.listdir(self.config.train_video_folder_path):
                video_path = os.path.join(self.config.train_video_folder_path, video)
                run_yolo_on_video(
                    video_path=video_path,
                    model_path=self.config.yolo_model_path,
                    conf=self.config.conf_threshold,
                )

    def remove_items_by_id(self, item_id_list: list[str]):
        """This function removes images and labels by their id."""
        remove_items_by_id(
            image_folder=os.path.join(self.config.train_dataset_path, "images"),
            label_folder=os.path.join(self.config.train_dataset_path, "labels"),
            item_id_list=item_id_list,
        )
        remove_items_by_id(
            image_folder=os.path.join(self.config.eval_dataset_path, "images"),
            label_folder=os.path.join(self.config.eval_dataset_path, "labels"),
            item_id_list=item_id_list,
        )

    def _process_dataset_split(
        self,
        split_name: str,
        video_folder: str | Path,
        dataset_path: str | Path,
    ) -> None:
        video_folder = Path(video_folder)
        dataset_path = Path(dataset_path)

        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        visualized_dir = dataset_path / "visualized_labels"

        logger.info(f"Extracting frames from {split_name} videos.")
        for video_path in video_folder.iterdir():
            if video_path.is_file():
                extract_frames(
                    video_path=str(video_path),
                    output_folder=str(images_dir),
                    frequency=self.config.extraction_framerate,
                )
            logger.info(f"Extracted frames from {video_path.name}.")

        logger.info(f"Labeling {split_name} images.")
        label_frames(
            image_folder=str(images_dir),
            output_label_folder=str(labels_dir),
            conf=0.25,
            max_sizes=self.config.max_sizes,
            min_sizes=self.config.min_sizes,
        )
        logger.info(f"Labeled {split_name} images.")

        logger.info(f"Removing images without objects in {split_name} dataset.")
        remove_images_without_objects(
            image_folder=str(images_dir),
            label_folder=str(labels_dir),
        )
        logger.info(f"Removed images without objects in {split_name} dataset.")

        logger.info(f"Visualizing {split_name} dataset labels.")
        visualize_lables(
            images_dir=str(images_dir),
            labels_dir=str(labels_dir),
            output_dir=str(visualized_dir),
        )
        logger.info(f"Visualized {split_name} dataset labels.")
