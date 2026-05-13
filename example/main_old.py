"""Main."""

import os

from vehicle_detector.helpers.labeling_helpers import (
    run_yolo_on_video,
)

if __name__ == "__main__":
    # video_path = os.path.join("videos", "train_C_640_360_24fps.mp4")
    # video_path = os.path.join("videos", "train_A_640_360_30fps.mp4")
    # video_path = os.path.join("videos", "train_B_640_360_24fps.mp4")
    run_yolo_on_video(video_path=os.path.join("eval_videos", "eval_360_640_30fps.mp4"))
    # video_path = os.path.join("train_videos", "train_D_640_360_30fps.mp4")
    # video_folder_name = "train_videos"

    # for video in os.listdir(video_folder_name):
    #     extract_frames(video_path=os.path.join(video_folder_name, video), output_folder="train_frames", frequency=5)

    max_sizes = {
        "train_A": [22, 10],
        "train_B": [12, 13],
        "train_C": [142, 159],
        "train_D": [50, 25],
    }
    # label_frames(image_folder="train_frames", output_label_folder="train_labels_filtered2", conf=0.25, max_sizes=max_sizes)
    # test_lables(
    #     images_dir=os.path.join("train_frames"),
    #     labels_dir=os.path.join("train_labels_filtered2"),
    #     output_dir=os.path.join("test_labels_filtered_output"),
    # )

    # extract_frames(
    #     video_path=os.path.join("videos", "train_A_640_360_30fps.mp4"),
    #     output_folder=os.path.join("frames_A", "train_A_640_360_30fps"),
    #     frequency=1.0,
    # )
    # predict_folder(image_folder=os.path.join("frames_C", "train_C_640_360_24fps"))
    # predict_folder(image_folder=os.path.join("frames_A", "train_A_640_360_30fps"))
    # # test_lables(
    # #     images_dir=os.path.join("frames_C", "train_C_640_360_24fps"),
    # #     labels_dir=os.path.join("frames_C", "train_C_640_360_24fps_auto_annotate_labels"),
    # #     output_dir=os.path.join("test_labels_output"),
    # # )
    # remove_images_without_objects(image_folder="train_frames", label_folder="train_labels")
    # remove_items_by_id(
    #     image_folder="train_frames",
    #     label_folder="train_labels",
    #     item_id_list=["train_C_640_360_24fps_000004", "train_C_640_360_24fps_000005", "train_C_640_360_24fps_000006",
    #                   "train_C_640_360_24fps_000007", "train_C_640_360_24fps_000008", "train_C_640_360_24fps_000009",
    #                   "train_C_640_360_24fps_000010", "train_C_640_360_24fps_000012", "train_C_640_360_24fps_000013",
    #                   "train_C_640_360_24fps_000015"]
    # )
