#!/usr/bin/env python3
"""
SAM3 point-based video segmentation using human annotations.

This script reads human annotation npz files, runs SAM3 video segmentation,
and saves the results (mask npz + visualization video).

Usage:
    python tools/test_sam3_point.py --username root --time 0
    python tools/test_sam3_point.py --username root --time 0 --no_save_video
    python tools/test_sam3_point.py --username root --time 0 --sam3_ckpt /path/to/sam3.pt
"""

import argparse
import json
import os
import pickle
import sys

import cv2
import numpy as np
import portalocker
import torch
import yaml
from tqdm import tqdm

# ─── Lists for post-processing ───────────────────────────────────────────────

UPDATE_VIDEO_LIST = []
FINISHED_VIDEO_LIST = []
HARD_SAMPLE_LIST = []
QUESTION_SAMPLE_LIST = []


# ─── Config helpers ───────────────────────────────────────────────────────────

def load_path_config(config_path="./config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    server_cfg = cfg["server"]
    root_dir = server_cfg["root_dir"]

    save_path_sam_tpl = server_cfg["save_path_sam_temp"].replace("/0/", "/{time}/")
    sam_mask_save_tpl = server_cfg["sam_mask_save_path"].replace("/0/", "/{time}/")
    sam_video_save_tpl = server_cfg["sam_video_save_path"].replace("/0/", "/{time}/")

    path_cfg = {
        "root_dir": root_dir,
        "user_sam_dir": os.path.join(root_dir, server_cfg["user_history_dir"], "sam"),
        "save_path_sam_tpl": save_path_sam_tpl,
        "sam_mask_save_tpl": sam_mask_save_tpl,
        "sam_video_save_tpl": sam_video_save_tpl,
        "video_dir": os.path.join(root_dir, server_cfg["video_dir"]),
        "error_log": os.path.join(root_dir, server_cfg["error_log"]),
        "no_annotation_sam": os.path.join(root_dir, server_cfg["no_annotation_sam"]),
        "has_annotation_sam": os.path.join(root_dir, server_cfg["has_annotation_sam"]),
    }
    return cfg, path_cfg


def resolve_path(root_dir, template, **kwargs):
    return os.path.join(root_dir, template.format(**kwargs))


def video_path_to_unique_name(video_path):
    """Extract filename without extension from a video path."""
    name = os.path.basename(video_path.strip())
    if name.endswith(".mp4"):
        name = name[:-4]
    return name


# ─── Helpers ──────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="SAM3 point-based video segmentation with existing annotations."
    )
    parser.add_argument("--username", type=str, required=True,
                        help="Username (reads user annotation list).")
    parser.add_argument("--time", type=int, default=0,
                        help="Annotation round (time). Default: 0")
    parser.add_argument("--config", type=str, default="./config/config.yaml",
                        help="Path to config.yaml. Default: ./config/config.yaml")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for SAM3 inference. Default: cuda:0")
    parser.add_argument("--sam3_ckpt", type=str, default=None,
                        help="Path to local SAM3 checkpoint. If not set, downloads from HuggingFace.")
    parser.add_argument("--no_save_video", action="store_true", default=False,
                        help="Disable saving visualization video.")
    return parser.parse_args()


def load_annotation(npz_path: str) -> dict:
    """Load annotation config from npz file (pickle-serialized)."""
    data = np.load(npz_path, allow_pickle=True)
    config = pickle.loads(data["arr_0"].item())
    return config


def extract_frames(video_path: str) -> np.ndarray:
    """Extract all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        raise RuntimeError(f"Failed to read any frames from {video_path}")
    return np.array(frames)


# ─── SAM3 model ───────────────────────────────────────────────────────────────

def build_sam3_model(device: str = "cuda:0", checkpoint_path: str = None):
    """Build SAM3 video model and return the tracker predictor."""
    from sam3.model_builder import build_sam3_video_model

    if checkpoint_path is not None:
        print(f"Building SAM3 video model from local checkpoint: {checkpoint_path}")
        sam3_model = build_sam3_video_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            device=device,
        )
    else:
        print("Building SAM3 video model (downloading checkpoint from HuggingFace)...")
        sam3_model = build_sam3_video_model(device=device)
    predictor = sam3_model.tracker
    predictor.backbone = sam3_model.detector.backbone
    print("SAM3 model loaded successfully.")
    return predictor


# ─── SAM3 segmentation core ──────────────────────────────────────────────────

def run_sam3_segmentation(
    predictor,
    video_path: str,
    model_config: dict,
    device: str = "cuda:0",
) -> np.ndarray:
    """
    Run SAM3 video segmentation using point prompts from model_config.

    Returns:
        masks: np.ndarray of shape (N_obj, T, 1, H, W), dtype bool
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    print(f"  Video: {width}x{height}, {num_frames} frames")

    inference_state = predictor.init_state(video_path=video_path)
    predictor.clear_all_points_in_video(inference_state)

    positive_points_dict = model_config["positive_points"]
    negative_points_dict = model_config["negative_points"]
    # Collect all object IDs across all annotated frames
    all_obj_ids = set()
    for frame_idx_str in positive_points_dict:
        for obj_id_str in positive_points_dict[frame_idx_str]:
            all_obj_ids.add(int(obj_id_str))
    all_obj_ids = sorted(all_obj_ids)
    if not all_obj_ids:
        raise ValueError("Annotation contains no positive point prompts")

    select_frames = model_config.get("select_frames", [])
    print(f"  Objects: {all_obj_ids}, Annotated frames: {select_frames}")

    with torch.inference_mode(), torch.autocast(device.split(":")[0], dtype=torch.bfloat16):
        for frame_idx_str in positive_points_dict:
            frame_idx = int(frame_idx_str)
            for obj_id_str in positive_points_dict[frame_idx_str]:
                obj_id = int(obj_id_str)

                pos_pts = positive_points_dict[frame_idx_str][obj_id_str]
                neg_pts = negative_points_dict.get(frame_idx_str, {}).get(obj_id_str, [])
                if len(pos_pts) == 0:
                    print(f"  Warning: No positive points for obj {obj_id} on frame {frame_idx}, skipping.")
                    continue

                all_points = list(pos_pts)
                if len(neg_pts) > 0:
                    all_points.extend(neg_pts)

                rel_points = [[x / width, y / height] for x, y in all_points]
                # Points are regrouped as positives followed by negatives, so
                # rebuild labels in that same order.
                sam3_labels = [1] * len(pos_pts) + [0] * len(neg_pts)

                points_tensor = torch.tensor(rel_points, dtype=torch.float32)
                labels_tensor = torch.tensor(sam3_labels, dtype=torch.int32)

                print(f"  Adding {len(all_points)} points (pos={len(pos_pts)}, neg={len(neg_pts)}) "
                      f"for obj {obj_id} on frame {frame_idx}")

                _, out_obj_ids, low_res_masks, video_res_masks = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points_tensor,
                    labels=labels_tensor,
                    clear_old_points=False,
                )

        # Propagate bidirectionally
        min_frame = min(int(f) for f in positive_points_dict.keys())
        video_segments = {}

        print("  Propagating forward...")
        for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min_frame,
            max_frame_num_to_track=num_frames,
            reverse=False,
            propagate_preflight=True,
        ):
            video_segments[frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(obj_ids)
            }

        if min_frame > 0:
            print("  Propagating backward...")
            for frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=min_frame,
                max_frame_num_to_track=num_frames,
                reverse=True,
                propagate_preflight=False,
            ):
                video_segments[frame_idx] = {
                    out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(obj_ids)
                }

    # Assemble masks: (N_obj, T, 1, H, W)
    masks_list = []
    for obj_id in all_obj_ids:
        obj_masks = []
        for t in range(num_frames):
            if t in video_segments and obj_id in video_segments[t]:
                mask = video_segments[t][obj_id]
                if mask.ndim == 2:
                    mask = mask[np.newaxis, :, :]
                elif mask.ndim == 3 and mask.shape[0] != 1:
                    mask = mask[:1]
            else:
                mask = np.zeros((1, height, width), dtype=bool)
            obj_masks.append(mask)
        obj_masks = np.stack(obj_masks, axis=0)
        masks_list.append(obj_masks)

    masks = np.stack(masks_list, axis=0).astype(bool)
    print(f"  Output masks shape: {masks.shape}")

    torch.cuda.empty_cache()
    return masks


# ─── Visualization ────────────────────────────────────────────────────────────

def save_visualization_video(
    video_frames: np.ndarray,
    masks: np.ndarray,
    output_path: str,
    fps: int = 20,
    alpha: float = 0.5,
):
    """
    Save a visualization video with mask overlay.

    Args:
        video_frames: (T, H, W, 3) BGR frames
        masks: (N_obj, T, 1, H, W) bool masks
    """
    import matplotlib.pyplot as plt

    n_obj = masks.shape[0]
    cmap = plt.get_cmap("tab10")
    colors = [np.array(cmap(i)[:3]) for i in range(n_obj)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    height, width = video_frames.shape[1:3]
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open MP4 writer for {output_path}")

    try:
        for t in range(video_frames.shape[0]):
            frame = video_frames[t].copy()
            for obj_id in range(n_obj):
                mask = masks[obj_id, t, 0]
                if mask.any():
                    color_overlay = (colors[obj_id] * 255).astype(np.uint8)
                    color_bgr = color_overlay[::-1]
                    colored = np.zeros_like(frame)
                    colored[:] = color_bgr
                    mask_3ch = mask[:, :, np.newaxis].repeat(3, axis=2)
                    frame = np.where(
                        mask_3ch,
                        ((1 - alpha) * frame + alpha * colored).astype(np.uint8),
                        frame,
                    )
                    ys, xs = np.where(mask)
                    if len(ys) > 0:
                        cy, cx = int(ys.mean()), int(xs.mean())
                        cv2.putText(
                            frame,
                            str(obj_id + 1),
                            (cx, cy),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            frame.shape[1] / 800,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

            writer.write(frame)
    finally:
        writer.release()
    print(f"  Visualization saved to {output_path}")


# ─── Batch processing ────────────────────────────────────────────────────────

def parse_and_save_results_sam3(line, predictor, path_cfg, user, time, device, save_video):
    """Process a single video line in batch mode."""
    root_dir = path_cfg["root_dir"]
    error_log = path_cfg["error_log"]
    video_name_no_ext = video_path_to_unique_name(line)

    video_save_path = resolve_path(root_dir, path_cfg["sam_video_save_tpl"],
                                   time=time, video_name=video_name_no_ext)
    sam_save_path = resolve_path(root_dir, path_cfg["sam_mask_save_tpl"],
                                 time=time, video_name=video_name_no_ext)
    model_config_path = resolve_path(root_dir, path_cfg["save_path_sam_tpl"],
                                     time=time, video_name=video_name_no_ext)

    # Load annotation config
    try:
        model_config = load_annotation(model_config_path)
    except Exception as e:
        print(f"Error in {model_config_path} because {e}", flush=True)
        os.makedirs(os.path.dirname(error_log), exist_ok=True)
        with open(error_log, "a") as f:
            f.write("\n" + model_config_path + "\t" + user + "\t" + str(e) + "\t" + line)
        return

    if "video_path" not in model_config:
        print(f"Error in {model_config_path}: no video_path", flush=True)
        with open(error_log, "a") as f:
            f.write("\n" + model_config_path + "\t" + user + "\tno video path")
        return

    if model_config.get("is_finished", False):
        print(model_config["video_path"] + " is finished", flush=True)
        if time >= 1:
            FINISHED_VIDEO_LIST.append(model_config["video_path"])
        return

    if model_config.get("is_hard_sample", False):
        hard_type = model_config.get("hard_sample_type", "unknown")
        if hard_type == "困难样本":
            print(model_config["video_path"] + " is hard sample", flush=True)
            if time >= 1:
                HARD_SAMPLE_LIST.append(model_config["video_path"])
            return
        elif hard_type == "问题样本":
            print(model_config["video_path"] + " is question sample", flush=True)
            if time >= 1:
                QUESTION_SAMPLE_LIST.append(model_config["video_path"])
            return

    # Resolve absolute video path
    abs_video_path = os.path.join(path_cfg["video_dir"], model_config["video_path"])
    model_config["origin_video_path"] = abs_video_path
    model_config["video_path"] = abs_video_path

    # Run SAM3 segmentation
    try:
        masks = run_sam3_segmentation(predictor, abs_video_path, model_config, device)
    except Exception as e:
        print(f"Error in {abs_video_path} because {e}", flush=True)
        with open(error_log, "a") as f:
            f.write("\n" + abs_video_path + "\t" + user + "\t" + str(e))
        return

    # Save mask npz
    os.makedirs(os.path.dirname(sam_save_path), exist_ok=True)
    np.savez_compressed(sam_save_path, masks=masks)
    print(f"  Mask saved to {sam_save_path}")

    # Save visualization video
    if save_video:
        os.makedirs(os.path.dirname(video_save_path), exist_ok=True)
        try:
            video_frames = extract_frames(model_config["origin_video_path"])
            if video_frames.ndim != 4 or video_frames.shape[0] == 0:
                print(f"Warning: failed to read video from {model_config['origin_video_path']}, skipping visualization.")
                UPDATE_VIDEO_LIST.append(video_save_path)
                return
            save_visualization_video(video_frames, masks, video_save_path)
        except Exception as e:
            print(f"Warning: visualization failed: {e}, skipping.")

    UPDATE_VIDEO_LIST.append(video_save_path)


def check_person_sam3(predictor, path_cfg, user_name, time, device, save_video):
    """Process all videos for a user at a given annotation round."""
    user_history_path = os.path.join(path_cfg["user_sam_dir"], user_name + ".txt")
    if time > 0:
        user_history_path = user_history_path.replace(".txt", f"_{time}.txt")

    if not os.path.exists(user_history_path):
        print(f"User {user_name} has no annotation at time {time}")
        sys.exit(1)

    with open(user_history_path) as f:
        user_ann_list = f.readlines()
    user_ann_list = [line.strip() for line in user_ann_list if line.strip()]

    print(f"Found {len(user_ann_list)} videos for user '{user_name}' at time {time}")

    for line in tqdm(user_ann_list, desc=f"Parsing {user_name} with time {time}"):
        parse_and_save_results_sam3(line, predictor, path_cfg, user_name, time, device, save_video)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    save_video = not args.no_save_video

    # Build SAM3 model
    predictor = build_sam3_model(device=args.device, checkpoint_path=args.sam3_ckpt)

    # Load config
    cfg, path_cfg = load_path_config(args.config)
    root_dir = path_cfg["root_dir"]

    # Ensure output directories exist
    os.makedirs(os.path.dirname(
        resolve_path(root_dir, path_cfg["sam_mask_save_tpl"], time=args.time, video_name="tmp")
    ), exist_ok=True)
    os.makedirs(os.path.dirname(
        resolve_path(root_dir, path_cfg["sam_video_save_tpl"], time=args.time, video_name="tmp")
    ), exist_ok=True)

    # Process all videos for the user
    check_person_sam3(predictor, path_cfg, args.username, args.time, args.device, save_video)

    # === Post-processing: update annotation pool JSONs ===
    # 1. Add processed videos to no_annotation_sam (queue for next round)
    add_file_number = 0
    with open(path_cfg["no_annotation_sam"], "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        no_annotation = json.load(f)
        for video_path in UPDATE_VIDEO_LIST:
            add_file_number += 1
            vname = video_path_to_unique_name(video_path)
            save_path = resolve_path(
                root_dir, path_cfg["save_path_sam_tpl"],
                time=args.time + 1, video_name=vname,
            )
            if video_path not in no_annotation.get(args.username, {}):
                no_annotation.setdefault(args.username, {})[video_path] = {
                    "anno_path": "",
                    "save_path": save_path,
                }
            print(f"Add {video_path} to no_annotation_sam")
        f.seek(0)
        f.truncate()
        json.dump(no_annotation, f)
        f.flush()
        os.fsync(f.fileno())
    print(f"Add {add_file_number} files to no_annotation_sam")

    # 2. Remove processed videos from has_annotation_sam
    delete_file_number = 0
    with open(path_cfg["has_annotation_sam"], "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        has_annotation = json.load(f)
        for video_path in UPDATE_VIDEO_LIST:
            if video_path in has_annotation.get(args.username, {}):
                del has_annotation[args.username][video_path]
                delete_file_number += 1
        f.seek(0)
        f.truncate()
        json.dump(has_annotation, f)
        f.flush()
        os.fsync(f.fileno())
    print(f"Delete {delete_file_number} files from has_annotation_sam")

    # 3. Clean up old entries and special cases from no_annotation_sam
    delete_file_number = 0
    finish_file_number = 0
    hard_file_number = 0
    question_file_number = 0
    with open(path_cfg["no_annotation_sam"], "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        no_annotation = json.load(f)
        for video_path in UPDATE_VIDEO_LIST:
            vname = video_path_to_unique_name(video_path)
            if args.time == 0:
                new_video_path = os.path.join(
                    path_cfg["video_dir"], os.path.basename(video_path)
                )
            else:
                new_video_path = resolve_path(
                    root_dir, path_cfg["sam_video_save_tpl"],
                    time=args.time - 1, video_name=vname,
                )
            if new_video_path in no_annotation.get(args.username, {}):
                del no_annotation[args.username][new_video_path]
                print(f"Delete {new_video_path} from no_annotation_sam")
                delete_file_number += 1
        for video_path in FINISHED_VIDEO_LIST:
            if video_path in no_annotation.get(args.username, {}):
                del no_annotation[args.username][video_path]
                finish_file_number += 1
        for video_path in HARD_SAMPLE_LIST:
            if video_path in no_annotation.get(args.username, {}):
                del no_annotation[args.username][video_path]
                hard_file_number += 1
        for video_path in QUESTION_SAMPLE_LIST:
            if video_path in no_annotation.get(args.username, {}):
                del no_annotation[args.username][video_path]
                question_file_number += 1
        f.seek(0)
        f.truncate()
        json.dump(no_annotation, f)
        f.flush()
        os.fsync(f.fileno())

    print(f"Delete {delete_file_number} files from no_annotation_sam")
    print(f"Finish {finish_file_number} files from no_annotation_sam")
    print(f"Hard {hard_file_number} files from no_annotation_sam")
    print(f"Question {question_file_number} files from no_annotation_sam")


if __name__ == "__main__":
    main()
