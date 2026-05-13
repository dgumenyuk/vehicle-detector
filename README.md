# Vehicle Detector

Vehicle detection pipeline built around YOLO.
Pipeline performs the following steps:

- extract frames from training and evaluation videos with a specified framerate i.e. `5 fps`;
- create YOLO-format vehicle labels based on the predictions of a pre-trained YOLO model, namely `yolo26s.yaml` from Ultralytics for the frames from training and evaluation videos; the generated labels are not fully accurate;
- anomalous labels are automatically removed if their dimensions do not match the expected size of the object bounding box;
- frames with no labels are removed from the training and evaluation datasets;
- train a YOLO model from the generated training dataset; model is trained from scratch, no pre-trained weights are used;
- evaluate model performance across distance bands, specifically `0-200 m` and `200-400 m` on the generated evaluation dataset;
- report precision, detection rate, false alarms per minute, and time to first detection


## Design choices

### Implementation
1. The vehicle detector is implemented as a python library. The main class is `VehicleDetector`, which takes a configuration yaml file as input and initializes the pipelines. The pipelines are implemented as separate classes: `LabelingPipeline`, `TrainingPipeline`, and `EvaluationPipeline`. This design allows for modularity and automated execution of the whole process: from dataset pseudo-labeling to model training and evaluation.
2. All the hyperparameters and paths are configured in a single yaml file. This allows for easy experimentation and reproducibility.

### Labeling and dataset creation
3. The labeling pipeline uses a pre-trained YOLO model to generate labels for the training and evaluation datasets. 
4. To improve the labelling quality, three measures are taken: first, the reasonable bounding box sizes are estimated by observing the predictions from the pseudo-labeling pipeline. Then, the generated labels are filtered by removing those with bounding box dimensions that do not match the expected size of the object bounding box. 
5. All the labels are visualized on the images for manual inspection. Images with anomalous labels are identified and removed from the dataset. 
6. Finally, images with no labels are removed from the datasets.

### Model training
7. The chosen object detection model is YOLOv26s from Ultralytics, which is a smaller version of the most recent YOLOv26 architecture.
8. Video size and thus the image resolution was chosen to be `640x360` as a compromise between computational efficiency and level of detail in the images. 
9. A new evaluation video was added, which contains more vehicles at farther distances.

### Evaluation metrics

10. The distance estimation function uses the apparent width of a detected object in the image to approximate its distance from the camera. The method is based on the pinhole camera model:

$$
Z = \frac{W \cdot f}{w}
$$

where:

- $Z$ is the estimated distance to the object in meters.
- $W$ is the assumed real-world width of the object in meters.
- $f$ is the camera focal length in pixels.
- $w$ is the apparent object width in the image, measured in pixels.

In the implementation:

```python
distance_m = reference_object_width_m * focal_length_px / box_width_px
```
The parameter `reference_object_width_m` is set to **4.5 meters** as an approximate real-world width reference.

The parameter `horizontal_fov_degrees` is set to **90 degrees** to approximate the horizontal field of view of the camera, selected based on the common FOV values for dash cameras.

Since the focal length in pixels is not directly known, it is computed from the image width and the horizontal field of view:

$$
f = \frac{\text{image width}}{2 \tan(\text{FOV}/2)}
$$
## Execution

### Installation

THis priject was tested on Windows and Ubuntu with python vesrions 3.12 - 3.14. Recommended python version is 3.14.
Main dependencies are listed in `pyproject.toml`, including `ultralytics`, `opencv-python`, `numpy`, `pydantic`, and `pyyaml`.

We recommend using `uv` for dependency management. Install with:

```bash
uv sync --all-groups
```
Pip or another package manager (e.g. conda) can also be used to install the package:

```bash
pip install .
```
Activate your virtual environment with:

```bash
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate  # Windows
```
Run the example pipeline with:

```bash
python example/main.py
```
Model training step is ommitted by default from the pipeline, as without good GPU it can take a long time. To include it, uncomment the corresponding lines in `example/main.py`. It is also recommended to ensure that the pytorch installation is compatible with the GPU and CUDA version on your machine. You can find the compatible versions on the [PyTorch website](https://pytorch.org/get-started/locally/).

We trained the model using the google collab, here is the [notebook link](https://colab.research.google.com/drive/1qq6bIQVQL3RYfCQWHDz4uGJAh3JYTa7-?usp=sharing) with the executable example. Make sure to connect to a runtime with GPU. 


### Configuration

Edit [example/pipeline_config.yaml](example/pipeline_config.yaml) for configuring custom pipiline parameters.
By defaut, the following parameters are set:

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
  image_frequency_fps: 5.0
  reference_object_width_m: 4.5
  horizontal_fov_degrees: 90.0
```

## Workflow

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

`N_frames` is the total number of evaluated image frames.



## Results

1. Examples of model performance on the validation data (from the train videos) are shown in `trained_models` folder.
This folder also contains the trained model weights and the training logs.
The achieved performance on the validation data is around 0.82 mAP@0.5.
2. Performance of the model on the evaluation videos is shown in `demo` folder (recordings with model predictions).
They can be reproduced by running this step of the evaluation pipeline:

```python
evaluation_pipeline: EvaluationPipeline = vehicle_detector.evaluation_pipeline
evaluation_pipeline.visualize_predictions(save=True)
```

3. The evaluation metrics:

| Distance range | Precision | Detection rate | False alarms / min | Time to first detection | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0–200 m | 0.240 | 0.113 | 287.973 | 1.000 s | 136 | 431 | 1069 |
| 200–400 m | 1.000 | 0.007 | 0.000 | 0.200 s | 12 | 0 | 1633 |