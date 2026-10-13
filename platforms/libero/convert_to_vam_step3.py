"""
Convert regenerated LIBERO HDF5 files into the per-episode VAM directory
format expected by hdf5_dataset_libero_T.

Input layout (regenerate_mask.py output):
    <source_dir>/
        <task_name>_demo.hdf5          # multi-demo HDF5
    <source_dir>/<suite>_metainfo.json # task descriptions

Output layout:
    <output_dir>/
        <dataset_name>/                # e.g. "libero_spatial_robot"
            <task_folder>/             # e.g. "task_00000__open_the_top_drawer..."
                episode_0/
                    episode.hdf5               # observations/robot_states (T,9), actions (T,7)
                    cam_high.mp4               # front camera video
                    cam_left_wrist.mp4         # wrist camera video
                    cam_high_mask.mp4          # front mask video (RGB: red=object, blue=robot, black=bg)
                    cam_left_wrist_mask.mp4    # wrist mask video (RGB: red=object, blue=robot, black=bg)
                    instructions.json          # {"seen": ["task description"]}
                episode_1/
                    ...

Usage:
    python platforms/libero/convert_to_vam_step3.py \
        --source_dir data/libero_v1/libero_spatial_no_noops_mask \
        --output_dir mydata/libero_vam \
        --dataset_name libero_spatial_robot \
        --meta_info data/libero_v1/libero_spatial_no_noops_mask/libero_spatial_metainfo.json
"""

import argparse
import json
import os

import cv2
import h5py
import numpy as np
from tqdm import tqdm


def encode_video(frames: np.ndarray, output_path: str, fps: int = 10):
    """Encode (T, H, W, 3) uint8 RGB frames to an mp4 video file.

    Args:
        frames: numpy array of shape (T, H, W, 3), dtype uint8, RGB order.
        output_path: path to write the .mp4 file.
        fps: frames per second.
    """
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    for i in range(T):
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    writer.release()


def mask_to_rgb_frames(mask: np.ndarray) -> np.ndarray:
    """Convert (T, H, W) float mask to (T, H, W, 3) uint8 RGB frames.

    Colour mapping:
        1.0  (object of interest) → Red   (255, 0, 0)
        0.5  (robot arm)          → Blue  (0, 0, 255)
        0.0  (background)         → Black (0, 0, 0)

    Args:
        mask: float array with values in {0.0, 0.5, 1.0} (or continuous [0, 1]).

    Returns:
        (T, H, W, 3) uint8 RGB array.
    """
    T, H, W = mask.shape
    rgb = np.zeros((T, H, W, 3), dtype=np.uint8)
    # Object pixels (mask >= 0.75) → Red
    obj_mask = mask >= 0.75
    rgb[obj_mask, 0] = 255  # R channel
    # Robot pixels (0.25 <= mask < 0.75) → Blue
    robot_mask = (mask >= 0.25) & (mask < 0.75)
    rgb[robot_mask, 2] = 255  # B channel
    return rgb


def encode_mask_video(mask: np.ndarray, output_path: str, fps: int = 10):
    """Convert (T, H, W) float mask to RGB and encode as mp4 video.

    Args:
        mask: float array of shape (T, H, W) with values in {0.0, 0.5, 1.0}.
        output_path: path to write the .mp4 file.
        fps: frames per second.
    """
    rgb_frames = mask_to_rgb_frames(mask)
    encode_video(rgb_frames, output_path, fps=fps)


def convert_single_hdf5(source_path: str, output_dir: str, task_description: str, task_idx: int):
    """Convert one multi-demo HDF5 file into per-episode VAM directories.

    Args:
        source_path: path to the source HDF5 file (regenerate_mask output).
        output_dir: root output directory for this dataset.
        task_description: language instruction for this task.
        task_idx: task index for folder naming.
    """
    # Build task folder name: task_00000__task_description_with_underscores
    safe_name = task_description.replace(" ", "_")
    safe_name = "".join(c for c in safe_name if c.isalnum() or c in "_-")
    task_folder = f"task_{task_idx:05d}__{safe_name}"
    task_dir = os.path.join(output_dir, task_folder)

    with h5py.File(source_path, "r") as source_f:
        if "data" not in source_f:
            print(f"Warning: 'data' group not found in {source_path}. Skipping.")
            return 0

        demo_keys = sorted(
            (k for k in source_f["data"].keys() if k.startswith("demo_")),
            key=lambda key: int(key.rsplit("_", 1)[-1]),
        )
        converted = 0

        for demo_key in tqdm(demo_keys, desc=f"  Converting {os.path.basename(source_path)}"):
            demo = source_f["data"][demo_key]
            episode_id = demo_key.split("_")[-1]
            episode_dir = os.path.join(task_dir, f"episode_{episode_id}")
            os.makedirs(episode_dir, exist_ok=True)

            # --- 1. episode.hdf5: state and action (unchanged) ---
            actions = demo["actions"][:]          # (T, 7)
            robot_states = demo["robot_states"][:]  # (T, 9)

            hdf5_path = os.path.join(episode_dir, "episode.hdf5")
            with h5py.File(hdf5_path, "w") as dest_f:
                obs_grp = dest_f.create_group("observations")
                obs_grp.create_dataset("robot_states", data=robot_states.astype(np.float32))
                dest_f.create_dataset("actions", data=actions.astype(np.float32))

            # --- 2. cam_high.mp4: front camera video ---
            agentview_rgb = demo["obs"]["agentview_rgb"][:]  # (T, H, W, 3) uint8
            video_path = os.path.join(episode_dir, "cam_high.mp4")
            encode_video(agentview_rgb, video_path)

            # --- 3. cam_left_wrist.mp4: wrist camera video ---
            eye_in_hand_rgb = demo["obs"]["eye_in_hand_rgb"][:]  # (T, H, W, 3) uint8
            wrist_video_path = os.path.join(episode_dir, "cam_left_wrist.mp4")
            encode_video(eye_in_hand_rgb, wrist_video_path)

            # --- 4. cam_high_mask.mp4: front camera mask video (RGB) ---
            agentview_mask = demo["obs"]["agentview_obj_mask"][:]  # (T, H, W)
            agentview_mask_f = agentview_mask.astype(np.float32)
            # Normalize: if max > 1 (e.g. old uint8 0/128/255 format), scale to [0, 1]
            if agentview_mask_f.max() > 1.0:
                agentview_mask_f /= 255.0
            mask_video_path = os.path.join(episode_dir, "cam_high_mask.mp4")
            encode_mask_video(agentview_mask_f, mask_video_path)

            # --- 5. cam_left_wrist_mask.mp4: wrist camera mask video (RGB) ---
            wrist_mask = demo["obs"]["eye_in_hand_obj_mask"][:]  # (T, H, W)
            wrist_mask_f = wrist_mask.astype(np.float32)
            # Normalize: if max > 1 (e.g. old uint8 0/128/255 format), scale to [0, 1]
            if wrist_mask_f.max() > 1.0:
                wrist_mask_f /= 255.0
            wrist_mask_video_path = os.path.join(episode_dir, "cam_left_wrist_mask.mp4")
            encode_mask_video(wrist_mask_f, wrist_mask_video_path)

            # --- 6. instructions.json ---
            instr_path = os.path.join(episode_dir, "instructions.json")
            with open(instr_path, "w") as f:
                json.dump({"seen": [task_description]}, f, indent=2)

            converted += 1

    return converted


def main(args):
    print("Converting LIBERO dataset to VAM episode format")
    print(f"  Source dir:    {args.source_dir}")
    print(f"  Output dir:    {args.output_dir}")
    print(f"  Dataset name:  {args.dataset_name}")

    # Load meta info for task descriptions
    with open(args.meta_info, "r") as f:
        meta_info = json.load(f)
    print(f"  Found {len(meta_info)} tasks in meta info")

    dataset_output_dir = os.path.join(args.output_dir, args.dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)

    # Collect all HDF5 files in source directory
    all_hdf5 = []
    for root, _, files in os.walk(args.source_dir):
        for fname in files:
            if fname.endswith(".hdf5"):
                all_hdf5.append((root, fname))
    print(f"  Found {len(all_hdf5)} HDF5 files in source directory")

    total_converted = 0
    for task_idx, task_name in enumerate(meta_info.keys()):
        print(f"\n[Task {task_idx}] {task_name}")

        # Find matching HDF5 file
        matching_file = None
        for root, fname in all_hdf5:
            if task_name in fname.lower():
                matching_file = os.path.join(root, fname)
                break

        if matching_file is None:
            print(f"  Warning: No matching HDF5 file found. Skipping.")
            continue

        # Use the task_name (with spaces) as the instruction
        task_description = task_name.replace("_", " ")
        n = convert_single_hdf5(matching_file, dataset_output_dir, task_description, task_idx)
        total_converted += n
        print(f"  Converted {n} episodes")

    print(f"\n=== Conversion Summary ===")
    print(f"Total tasks: {len(meta_info)}")
    print(f"Total episodes converted: {total_converted}")
    print(f"Output directory: {dataset_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LIBERO HDF5 to VAM episode format.")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Directory containing regenerated HDF5 files from regenerate_mask.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory for converted dataset.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="libero_goal_robot",
        help="Dataset subfolder name (used as dataset_name in yaml config).",
    )
    parser.add_argument(
        "--meta_info",
        type=str,
        required=True,
        help="Path to the metainfo JSON file from regenerate_mask.py.",
    )
    args = parser.parse_args()
    main(args)
