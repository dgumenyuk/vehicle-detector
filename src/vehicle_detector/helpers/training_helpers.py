import os
import random
import shutil
from pathlib import Path

import yaml

from vehicle_detector.helpers import logger


def copy_files(file_list, subset, rgb_path, label_path, save_path):
    img_dst = os.path.join(save_path, subset, "images")
    lbl_dst = os.path.join(save_path, subset, "labels")

    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(lbl_dst, exist_ok=True)

    for img_file in file_list:
        name, ext = os.path.splitext(img_file)
        src_img = os.path.join(rgb_path, img_file)
        src_lbl = os.path.join(label_path, name + ".txt")

        # Copy image
        shutil.copy(src_img, os.path.join(img_dst, img_file))

        # Copy label if it exists
        if os.path.exists(src_lbl):
            shutil.copy(src_lbl, os.path.join(lbl_dst, name + ".txt"))
        else:
            logger.info(f"Warning: Label missing for {img_file}")


def train_test_split(
    image_path: str,
    label_path: str,
    save_path: str,
    train_percentage: float = 0.8,
    class_name: str = "vehicle",
) -> None:
    """Splits the dataset into a training and a test set."""

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(image_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    random.shuffle(image_files)

    train_count = round(len(image_files) * train_percentage)
    train_files = image_files[:train_count]
    val_files = image_files[train_count:]

    logger.info(f"Total images: {len(image_files)}")
    logger.info(f"Train set: {len(train_files)} images")
    logger.info(f"Val set: {len(val_files)} images")

    # Copy files
    copy_files(train_files, "train", image_path, label_path, save_path)
    copy_files(val_files, "val", image_path, label_path, save_path)

    # Optional: create dataset.yaml file
    dataset_yaml = {
        "path": str(Path(save_path).resolve()),
        "train": str(Path("train/images")),
        "val": str(Path("val/images")),
        "names": {0: class_name},
    }

    yaml_path = Path(save_path) / "dataset.yaml"

    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)

    print(f"Saved YAML to: {yaml_path.resolve()}")
