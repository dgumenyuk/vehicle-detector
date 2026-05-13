# Vehicle Detector

Vehicle detection pipeline built around YOLO. The project can:

- extract frames from training and evaluation videos
- create YOLO-format vehicle labels
- train a YOLO model
- evaluate detections across distance bands, currently `0-200 m` and `200-400 m`
- report precision, detection rate, false alarms per minute, and time to first detection


## Installation

From the project root:

```powershell
uv sync --all-groups
```

Or with pip:

```powershell
pip install .
```

Main dependencies are listed in `pyproject.toml`, including `ultralytics`, `opencv-python`, `numpy`, `pydantic`, and `pyyaml`.

## Configuration

Edit [example/pipeline_config.yaml](example/pipeline_config.yaml) before running a pipeline.

Important sections:

```yaml
labeling:
  train_video_folder_path: "train_videos"
  train_dataset_path: "train_dataset"
  eval_video_folder_path: "eval_videos"
  eval_dataset_path: "eval_dataset"
  conf_threshold: 0.25
  extraction_framerate: 5

training:
  train_dataset_path: "train_dataset"
  model_save_path: "trained_models"
  yolo_model_config: "yolo26s.yaml"

evaluation:
  eval_dataset_path: "eval_dataset"
  eval_video_folder_path: "eval_videos"
  model_path: "trained_models/your_model.pt"
  conf_threshold: 0.2
  iou_threshold: 0.5
  image_frequency_fps: 1.0
  reference_object_width_m: 4.5
  horizontal_fov_degrees: 90.0
```

## Running

The example runner is [example/main.py](example/main.py). Uncomment the pipeline you want to run.

Run it with:

```powershell
python example/main.py
```

Typical workflow:

1. Put training videos in `train_videos/` and evaluation videos in `eval_videos/`.
2. Run the labeling pipeline to extract frames and generate labels.
3. Inspect generated labels in `train_dataset/visualized_labels/` and `eval_dataset/visualized_labels/`.
4. Run the training pipeline.
5. Update `evaluation.model_path` in `pipeline_config.yaml` to point to the trained `.pt` file.
6. Run the evaluation pipeline.

## Evaluation Metrics

The evaluation pipeline compares predicted boxes against ground-truth boxes using IoU.

A prediction is counted as:

- `TP`: prediction overlaps a ground-truth box with IoU greater than or equal to `iou_threshold`
- `FN`: ground-truth box has no matching prediction
- `FP`: prediction has no matching ground-truth box

The reported metrics are:

```text
Detection rate = TP / (TP + FN)
Precision      = TP / (TP + FP)
False alarms   = FP * 60 * image_frequency_fps / N_frames
Time to first detection = (first_detection_frame - first_gt_frame) / image_frequency_fps
```

`N_frames` is the number of evaluated image frames.

## Distance Estimation

Evaluation separates objects into distance bands. The distance is estimated from the width of the bounding box in pixels.

The code assumes:

- the real-world width of the object is known, configured as `reference_object_width_m`
- the camera horizontal field of view is known, configured as `horizontal_fov_degrees`
- objects farther away appear smaller in the image
- the measured box width is a reasonable approximation of the object's visible width

First, the camera focal length is estimated in pixels:

```text
focal_length_px = image_width_px / (2 * tan(horizontal_fov_radians / 2))
```

Then object distance is estimated with the pinhole camera relationship:

```text
distance_m = reference_object_width_m * focal_length_px / box_width_px
```

Where:

- `reference_object_width_m` is the assumed real object width, for example car width or vehicle length depending on how boxes are viewed
- `box_width_px` is `x2 - x1` from the bounding box
- `image_width_px` is the width of the frame
- `horizontal_fov_radians` is `horizontal_fov_degrees` converted to radians

Example:

```text
image_width_px = 640
horizontal_fov_degrees = 90
reference_object_width_m = 4.5
box_width_px = 72

focal_length_px = 640 / (2 * tan(90 / 2)) = 320
distance_m = 4.5 * 320 / 72 = 20 m
```

After estimating distance, the object is assigned to a band:

```text
0 <= distance < 200    -> 0-200 m
200 <= distance < 400  -> 200-400 m
```

This is an approximation. It is sensitive to the chosen reference width, camera FOV, camera calibration, object orientation, and whether the box width truly represents the physical dimension being used.

## Notes

- The project currently treats vehicles as a single class.
- Evaluation reads images from `eval_dataset/images` and labels from `eval_dataset/labels`.
- The model path in `pipeline_config.yaml` must point to an existing YOLO `.pt` file.
- For better distance estimates, calibrate the camera or tune `reference_object_width_m` and `horizontal_fov_degrees` using objects at known distances.
