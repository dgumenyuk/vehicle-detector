import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from vehicle_detector.helpers import logger


def run_yolo_on_video(
    video_path: str,
    model_path: str | None = None,
    save_video: bool = False,
    output_path: str | Path = "annotated_output.mp4",
    conf: float = 0.25,
    show_label: bool = True,
) -> None:

    if model_path is None:
        model = YOLO("yolo26x.pt")
    else:
        model = YOLO(model_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (frame_width, frame_height),
        )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        result = results[0]

        annotated = frame.copy()

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                box_width = x2 - x1
                box_height = y2 - y1

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )
                if show_label:
                    label = f"{box_width}x{box_height} px"
                    # Draw label background
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - 25),
                        (x1 + 110, y1),
                        (0, 255, 0),
                        -1,
                    )

                    # Draw size text
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2,
                    )

        cv2.imshow("detections", annotated)

        if save_video and writer is not None:
            writer.write(annotated)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()


def extract_frames(video_path: str, output_folder: str, frequency: float = 1.0) -> None:
    """
    Extract frames from a video at a given frequency.

    Args:
        video_path: Path to input video
        output_folder: Folder to save extracted frames
        frequency: Frames per second to extract
                   (e.g. 1.0 = 1 frame/sec)
    """

    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError("Could not determine FPS")

    # Number of frames to skip between saves
    frame_interval = int(round(fps / frequency))
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:
            frame_name = f"{video_name}_{saved_id:06d}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)

            saved_id += 1

        frame_id += 1

    cap.release()


def box_too_big(
    image_path: str,
    max_sizes: dict[str, list[float]] | None,
    width: float,
    height: float,
) -> bool:
    if max_sizes is None:
        return False

    for name, (max_width, max_height) in max_sizes.items():
        if name in image_path:
            return width > max_width or height > max_height

    return False


def box_too_small(
    image_path: str,
    min_sizes: dict[str, list[float]] | None,
    width: float,
    height: float,
) -> bool:
    if min_sizes is None:
        return False

    for name, (min_width, min_height) in min_sizes.items():
        if name in image_path:
            return width < min_width or height < min_height

    return False


def label_frames(
    image_folder: str | Path,
    output_label_folder: str | Path = "labels",
    target_class_id: int | None = None,
    conf: float = 0.25,
    max_sizes: dict[str, list[float]] | None = None,
    min_sizes: dict[str, list[float]] | None = None,
) -> None:
    model = YOLO("yolo26x.pt")

    image_folder = Path(image_folder)
    output_label_folder = Path(output_label_folder)
    output_label_folder.mkdir(parents=True, exist_ok=True)

    image_files = [
        f
        for f in image_folder.iterdir()
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    for i, image_path in enumerate(image_files):
        image = cv2.imread(str(image_path))
        if image is None:
            logger.info(f"[!] Skipping invalid image: {image_path}")
            continue

        img_h, img_w = image.shape[:2]
        results = model.predict(
            str(image_path),
            imgsz=640,
            conf=conf,
            save=False,
            verbose=False,
        )

        result = results[0]
        label_path = output_label_folder / f"{image_path.stem}.txt"

        lines = []
        class_id = 0  # Assuming single class i.e. "vehicle". Adjust if multiple classes are used.
        if result.boxes is not None:
            for box in result.boxes:
                # Keep only selected class if target_class_id is given
                if target_class_id is not None and class_id != target_class_id:
                    continue

                x_center, y_center, width, height = box.xywhn[0].tolist()

                if not box_too_big(
                    str(image_path), max_sizes, width * img_w, height * img_h
                ) and not box_too_small(
                    str(image_path), min_sizes, width * img_w, height * img_h
                ):
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    )

        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        if i % 20 == 0:
            logger.info(f"Labeled {i}/{len(image_files)} images.")


def load_labels(label_path, img_w, img_h):
    bboxes = []
    if not os.path.exists(label_path):
        logger.info("Labels not found")
        return bboxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # skip invalid lines

            x_center, y_center, width, height = map(float, parts[1:])

            # Convert normalized coordinates to pixel values
            x_center *= img_w
            y_center *= img_h
            width *= img_w
            height *= img_h
            clss_id = int(parts[0])

            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            bboxes.append((clss_id, (x1, y1, x2, y2)))

    return bboxes


def visualize_lables(images_dir: str, labels_dir: str, output_dir: str) -> None:
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    # Create output folder if not exists
    os.makedirs(output_dir, exist_ok=True)

    # === Helper: Load YOLO-format labels and convert to bounding boxes ===

    # === Process All Images ===
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(image_exts):
            continue

        image_path = os.path.join(images_dir, fname)
        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        output_path = os.path.join(output_dir, fname)

        image = cv2.imread(image_path)
        if image is None:
            logger.info(f"[!] Skipping invalid image: {image_path}")
            continue

        img_h, img_w = image.shape[:2]
        bboxes = load_labels(label_path, img_w, img_h)

        # Draw bounding boxes
        for cls, (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cls}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

        # Save annotated image
        cv2.imwrite(output_path, image)


def remove_images_without_objects(
    image_folder: str,
    label_folder: str,
) -> None:
    """
    Deletes images and labels for which no object is detected.

    Assumes YOLO format labels:
    - one .txt file per image
    - empty .txt file => no objects

    Parameters
    ----------
    image_folder : str
        Path to image directory.

    label_folder : str
        Path to YOLO label directory.

    delete_empty_label_only : bool
        If True:
            deletes only empty label files.
        If False:
            deletes both image and label.
    """

    image_folder: Path = Path(image_folder)
    label_folder: Path = Path(label_folder)

    image_extensions = [".jpg", ".jpeg", ".png", ".JPG"]

    removed_count = 0

    for image_path in image_folder.iterdir():
        if image_path.suffix not in image_extensions:
            continue

        label_path = label_folder / f"{image_path.stem}.txt"

        # No label file
        if not label_path.exists():
            logger.info(f"No label file for: {image_path.name}")
            continue

        # Check if label file is empty
        with open(label_path) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) == 0:
            # Delete label
            os.remove(label_path)
            # Delete image
            os.remove(image_path)
            removed_count += 1

    logger.info(f"\nRemoved {removed_count} empty samples.")


def remove_items_by_id(
    image_folder: str, label_folder: str, item_id_list: list[str]
) -> None:
    """
    Deletes image and label by item ID.

    Parameters
    ----------
    image_folder : str
        Path to image directory.

    label_folder : str
        Path to YOLO label directory.

    item_id : str
        ID of the item to delete (e.g. "video_000123_000045").
    """

    image_extensions = [".jpg", ".jpeg", ".png", ".JPG"]

    for item_id in item_id_list:
        for ext in image_extensions:
            image_path = Path(image_folder) / f"{item_id}{ext}"
            if image_path.exists():
                os.remove(image_path)
                logger.info(f"Deleted image: {image_path.name}")
                break

        label_path = Path(label_folder) / f"{item_id}.txt"
        if label_path.exists():
            os.remove(label_path)
            logger.info(f"Deleted label: {label_path.name}")
