from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel
from ultralytics import YOLO

from vehicle_detector.helpers import logger
from vehicle_detector.helpers.labeling_helpers import run_yolo_on_video


class EvaluationPipelineConfig(BaseModel):
    eval_dataset_path: str
    model_path: str
    eval_video_folder_path: str

    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    image_frequency_fps: float = 5.0

    reference_object_width_m: float = 2
    horizontal_fov_degrees: float = 90


class EvaluationPipeline:
    def initialize(self, config):
        self.config = EvaluationPipelineConfig(**config)
        logger.info("Initialized evaluation pipeline.")

    def visualize_predictions(self, save: bool = False) -> None:
        video_dir = Path(self.config.eval_video_folder_path)
        save_dir = Path(self.config.eval_dataset_path) / "prediction_visualizations"
        save_dir.mkdir(parents=True, exist_ok=True)
        for video_path in video_dir.iterdir():
            if not video_path.is_file():
                continue

            run_yolo_on_video(
                video_path=str(video_path),
                model_path=self.config.model_path,
                conf=self.config.conf_threshold,
                save_video=save,
                output_path=save_dir / f"{video_path.stem}.mp4",
                show_label=False,
            )

    def run(self):
        model = YOLO(self.config.model_path)

        dataset_path = Path(self.config.eval_dataset_path)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"

        stats = {
            "0-200 m": {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "first_gt_frame": None,
                "first_detection_frame": None,
            },
            "200-400 m": {
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "first_gt_frame": None,
                "first_detection_frame": None,
            },
        }

        image_paths = sorted(images_dir.glob("*.jpg"))

        for frame_id, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image_height, image_width = image.shape[:2]

            gt_boxes = self.load_ground_truth(
                labels_dir / f"{image_path.stem}.txt",
                image_width,
                image_height,
            )

            pred_boxes = self.run_model(model, image)

            matched_predictions = set()

            for gt_box in gt_boxes:
                gt_bbox = gt_box["bbox"]

                band = self.get_distance_band(gt_bbox, image_width)

                if band is None:
                    continue

                if stats[band]["first_gt_frame"] is None:
                    stats[band]["first_gt_frame"] = frame_id

                best_iou = 0
                best_pred_id = None

                for pred_id, pred_box in enumerate(pred_boxes):
                    if pred_id in matched_predictions:
                        continue

                    iou = self.compute_iou(gt_bbox, pred_box["bbox"])

                    if iou > best_iou:
                        best_iou = iou
                        best_pred_id = pred_id

                if best_iou >= self.config.iou_threshold:
                    stats[band]["TP"] += 1
                    matched_predictions.add(best_pred_id)

                    if stats[band]["first_detection_frame"] is None:
                        stats[band]["first_detection_frame"] = frame_id
                else:
                    stats[band]["FN"] += 1

            for pred_id, pred_box in enumerate(pred_boxes):
                if pred_id in matched_predictions:
                    continue

                band = self.get_distance_band(pred_box["bbox"], image_width)

                if band is not None:
                    stats[band]["FP"] += 1

        self.print_results(stats, n_frames=len(image_paths))

    def load_ground_truth(self, label_path, image_width, image_height):
        boxes = []

        if not label_path.exists():
            return boxes

        with open(label_path) as f:
            for line in f:
                class_id, xc, yc, bw, bh = map(float, line.split())

                x1 = (xc - bw / 2) * image_width
                y1 = (yc - bh / 2) * image_height
                x2 = (xc + bw / 2) * image_width
                y2 = (yc + bh / 2) * image_height

                boxes.append(
                    {
                        "class_id": int(class_id),
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        return boxes

    def run_model(self, model, image):
        result = model.predict(
            image,
            conf=self.config.conf_threshold,
            verbose=False,
        )[0]

        boxes = []

        for box in result.boxes:
            boxes.append(
                {
                    "class_id": int(box.cls.item()),
                    "bbox": box.xyxy.cpu().numpy()[0],
                }
            )

        return boxes

    def get_distance_band(self, bbox, image_width):
        distance_m = self.estimate_distance(bbox, image_width)

        if 0 <= distance_m < 200:
            return "0-200 m"

        if 200 <= distance_m < 400:
            return "200-400 m"

        return None

    def estimate_distance(self, bbox, image_width):
        box_width_px = max(1.0, bbox[2] - bbox[0])

        fov_rad = np.deg2rad(self.config.horizontal_fov_degrees)

        focal_length_px = image_width / (2 * np.tan(fov_rad / 2))

        distance_m = (
            self.config.reference_object_width_m * focal_length_px / box_width_px
        )

        return distance_m

    def compute_iou(self, box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
        area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])

        union = area_a + area_b - intersection

        if union == 0:
            return 0

        return intersection / union

    def print_results(self, stats, n_frames):
        for band, values in stats.items():
            TP = values["TP"]
            FP = values["FP"]
            FN = values["FN"]
            frames = n_frames

            precision = TP / (TP + FP) if TP + FP > 0 else 0
            detection_rate = TP / (TP + FN) if TP + FN > 0 else 0

            if frames > 0:
                false_alarms_per_min = (
                    FP * 60 * self.config.image_frequency_fps / frames
                )
            else:
                false_alarms_per_min = 0

            first_gt = values["first_gt_frame"]
            first_detection = values["first_detection_frame"]

            if first_gt is not None and first_detection is not None:
                time_to_first_detection = (
                    first_detection - first_gt
                ) / self.config.image_frequency_fps
            else:
                time_to_first_detection = None

            print()
            print(band)
            print(f"Precision: {precision:.3f}")
            print(f"Detection rate: {detection_rate:.3f}")
            print(f"False alarms / min: {false_alarms_per_min:.3f}")

            if time_to_first_detection is None:
                print("Time to first detection: not detected")
            else:
                print(f"Time to first detection: {time_to_first_detection:.3f} s")

            print(f"TP: {TP}")
            print(f"FP: {FP}")
            print(f"FN: {FN}")
