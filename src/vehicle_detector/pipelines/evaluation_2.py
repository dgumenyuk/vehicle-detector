import math
from dataclasses import dataclass
from pathlib import Path

import cv2
from pydantic import BaseModel
from ultralytics import YOLO

from vehicle_detector.helpers import logger
from vehicle_detector.helpers.labeling_helpers import run_yolo_on_video
from vehicle_detector.pipelines import BasePipeline


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @property
    def width(self) -> float:
        return max(self.x2 - self.x1, 1.0)


@dataclass
class BandMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    first_detection_frame: int | None = None


class EvaluationPipelineConfig(BaseModel):
    eval_dataset_path: str
    eval_video_folder_path: str
    model_path: str | None = None
    predictions_labels_path: str | None = None
    output_report_path: str = "evaluation_metrics.md"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    image_frequency_fps: float = 1.0
    reference_object_length_m: float = 4.5
    horizontal_fov_degrees: float = 90.0
    distance_bands_m: tuple[tuple[float, float], ...] = (
        (0.0, 200.0),
        (200.0, 400.0),
    )


class EvaluationPipeline(BasePipeline):
    def initialize(self, config):
        self.config = EvaluationPipelineConfig(**config)
        logger.info("Initialized evaluation pipeline.")

    def visualize_predictions(self) -> None:
        video_dir = Path(self.config.eval_video_folder_path)
        for video_path in video_dir.iterdir():
            if not video_path.is_file():
                continue

            run_yolo_on_video(
                video_path=str(video_path),
                model_path=self.config.model_path,
                conf=self.config.conf_threshold,
            )

    def run(self) -> None:
        logger.info("Running evaluation pipeline.")

        dataset_path = Path(self.config.eval_dataset_path)
        images_dir = dataset_path / "images"
        labels_dir = dataset_path / "labels"
        predictions_dir = self._prepare_predictions(dataset_path, images_dir)

        image_paths = self._image_paths(images_dir)
        metrics = {
            self._band_label(band): BandMetrics()
            for band in self.config.distance_bands_m
        }

        for frame_index, image_path in enumerate(image_paths):
            image = cv2.imread(str(image_path))
            if image is None:
                logger.info(f"Skipping invalid image: {image_path}")
                continue

            img_h, img_w = image.shape[:2]
            gt_boxes = self._read_yolo_file(
                labels_dir / f"{image_path.stem}.txt",
                img_w,
                img_h,
            )
            predicted_boxes = self._read_yolo_file(
                predictions_dir / f"{image_path.stem}.txt",
                img_w,
                img_h,
            )

            self._count_frame_metrics(
                metrics,
                gt_boxes,
                predicted_boxes,
                img_w,
                frame_index,
            )

        report = self._format_report(metrics, num_frames=len(image_paths))
        output_path = Path(self.config.output_report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)

        logger.info("\n" + report)
        logger.info(f"Saved evaluation report to {output_path}")

    def _prepare_predictions(self, dataset_path: Path, images_dir: Path) -> Path:
        if self.config.predictions_labels_path is not None:
            return Path(self.config.predictions_labels_path)

        predictions_dir = dataset_path / "evaluation_predictions" / "labels"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(self.config.model_path or "yolo26x.pt")
        for image_path in self._image_paths(images_dir):
            results = model.predict(
                str(image_path),
                imgsz=640,
                conf=self.config.conf_threshold,
                save=False,
                verbose=False,
            )
            lines = []
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} "
                        f"{width:.6f} {height:.6f} {confidence:.6f}"
                    )

            (predictions_dir / f"{image_path.stem}.txt").write_text("\n".join(lines))

        return predictions_dir

    def _image_paths(self, images_dir: Path) -> list[Path]:
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        return sorted(
            path
            for path in images_dir.iterdir()
            if path.suffix.lower() in image_extensions
        )

    def _read_yolo_file(self, label_path: Path, img_w: int, img_h: int) -> list[Box]:
        if not label_path.exists():
            return []

        boxes = []
        with label_path.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) not in {5, 6}:
                    continue

                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                confidence = float(parts[5]) if len(parts) == 6 else 1.0

                x_center *= img_w
                y_center *= img_h
                width *= img_w
                height *= img_h

                boxes.append(
                    Box(
                        class_id=class_id,
                        x1=x_center - width / 2,
                        y1=y_center - height / 2,
                        x2=x_center + width / 2,
                        y2=y_center + height / 2,
                        confidence=confidence,
                    )
                )

        return boxes

    def _count_frame_metrics(
        self,
        metrics: dict[str, BandMetrics],
        gt_boxes: list[Box],
        predicted_boxes: list[Box],
        img_w: int,
        frame_index: int,
    ) -> None:
        gt_bands = [self._box_distance_band(box, img_w) for box in gt_boxes]
        matched_gt_indexes = set()

        for predicted_box in sorted(
            predicted_boxes,
            key=lambda box: box.confidence,
            reverse=True,
        ):
            gt_index = self._find_best_match(predicted_box, gt_boxes, matched_gt_indexes)

            if gt_index is None:
                band = self._box_distance_band(predicted_box, img_w)
                if band is not None:
                    metrics[band].fp += 1
                continue

            matched_gt_indexes.add(gt_index)
            band = gt_bands[gt_index]
            if band is None:
                continue

            metrics[band].tp += 1
            if metrics[band].first_detection_frame is None:
                metrics[band].first_detection_frame = frame_index

        for gt_index, band in enumerate(gt_bands):
            if gt_index not in matched_gt_indexes and band is not None:
                metrics[band].fn += 1

    def _find_best_match(
        self,
        predicted_box: Box,
        gt_boxes: list[Box],
        matched_gt_indexes: set[int],
    ) -> int | None:
        best_gt_index = None
        best_iou = 0.0

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt_indexes:
                continue
            if gt_box.class_id != predicted_box.class_id:
                continue

            iou = self._iou(predicted_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index

        if best_iou >= self.config.iou_threshold:
            return best_gt_index
        return None

    def _box_distance_band(self, box: Box, img_w: int) -> str | None:
        distance_m = self._estimate_distance_m(box.width, img_w)

        for band in self.config.distance_bands_m:
            low_m, high_m = band
            if low_m <= distance_m < high_m:
                return self._band_label(band)

        return None

    def _estimate_distance_m(self, box_width_px: float, img_w: int) -> float:
        fov_radians = math.radians(self.config.horizontal_fov_degrees)
        focal_length_px = img_w / (2 * math.tan(fov_radians / 2))
        return self.config.reference_object_length_m * focal_length_px / box_width_px

    def _iou(self, first: Box, second: Box) -> float:
        overlap_x1 = max(first.x1, second.x1)
        overlap_y1 = max(first.y1, second.y1)
        overlap_x2 = min(first.x2, second.x2)
        overlap_y2 = min(first.y2, second.y2)

        overlap_w = max(0.0, overlap_x2 - overlap_x1)
        overlap_h = max(0.0, overlap_y2 - overlap_y1)
        overlap_area = overlap_w * overlap_h

        first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
        second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
        union_area = first_area + second_area - overlap_area

        if union_area <= 0:
            return 0.0
        return overlap_area / union_area

    def _format_report(
        self,
        metrics: dict[str, BandMetrics],
        num_frames: int,
    ) -> str:
        band_labels = [self._band_label(band) for band in self.config.distance_bands_m]

        rows = [
            ("Detection rate TP / (TP + FN)", self._detection_rate),
            ("Precision TP / (TP + FP)", self._precision),
            ("False alarms / min FP x 60 / N_frames", self._false_alarms_per_min),
            ("Time to first detection seconds", self._time_to_first_detection),
        ]

        lines = [
            "# Evaluation metrics across distance bands",
            "",
            "Assumption: distance is estimated from the GT box width, car length, "
            "image width, and camera horizontal FOV.",
            "",
            "| Metric | " + " | ".join(band_labels) + " |",
            "| --- | " + " | ".join("---" for _ in band_labels) + " |",
        ]

        for row_name, metric_function in rows:
            values = [
                metric_function(metrics[band_label], num_frames)
                for band_label in band_labels
            ]
            lines.append(f"| {row_name} | " + " | ".join(values) + " |")

        return "\n".join(lines)

    def _detection_rate(self, metrics: BandMetrics, num_frames: int) -> str:
        return self._format_ratio(metrics.tp, metrics.tp + metrics.fn)

    def _precision(self, metrics: BandMetrics, num_frames: int) -> str:
        return self._format_ratio(metrics.tp, metrics.tp + metrics.fp)

    def _false_alarms_per_min(self, metrics: BandMetrics, num_frames: int) -> str:
        if num_frames == 0:
            return "0.000"
        value = metrics.fp * 60 / num_frames
        return f"{value:.3f}"

    def _time_to_first_detection(self, metrics: BandMetrics, num_frames: int) -> str:
        if metrics.first_detection_frame is None:
            return "N/A"
        value = metrics.first_detection_frame / self.config.image_frequency_fps
        return f"{value:.3f}"

    def _format_ratio(self, numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "0.000"
        return f"{numerator / denominator:.3f}"

    def _band_label(self, band: tuple[float, float]) -> str:
        low_m, high_m = band
        return f"{low_m:g}-{high_m:g} m"
