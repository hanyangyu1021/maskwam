import pickle
import numpy as np
import portalocker
import cv2, os, yaml
import json

from tqdm import tqdm
import argparse

from test_sam3_point import (
    build_sam3_model,
    run_sam3_segmentation,
    save_visualization_video,
)

UPDATE_VIDEO_LIST = []
FINISHED_VIDEO_LIST = []
HARD_SAMPLE_LIST = []
QUESTION_SAMPLE_LIST = []


def load_path_config(config_path="./config/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    server_cfg = cfg["server"]
    root_dir = server_cfg["root_dir"]

    # Derive timed path templates (replace hardcoded /0/ with /{time}/)
    save_path_sam_tpl = server_cfg["save_path_sam_temp"].replace("/0/", "/{time}/")
    sam_mask_save_tpl = server_cfg["sam_mask_save_path"].replace("/0/", "/{time}/")
    sam_video_save_tpl = server_cfg["sam_video_save_path"].replace("/0/", "/{time}/")

    path_cfg = {
        "root_dir": root_dir,
        "user_sam_dir": os.path.join(root_dir, server_cfg["user_history_dir"], "sam"),
        # Templates: call .format(time=X, video_name=Y), then join with root_dir
        "save_path_sam_tpl": save_path_sam_tpl,
        "sam_mask_save_tpl": sam_mask_save_tpl,
        "sam_video_save_tpl": sam_video_save_tpl,
        # Absolute paths
        "video_dir": os.path.join(root_dir, server_cfg["video_dir"]),
        "error_log": os.path.join(root_dir, server_cfg["error_log"]),
        "no_annotation_sam": os.path.join(root_dir, server_cfg["no_annotation_sam"]),
        "has_annotation_sam": os.path.join(root_dir, server_cfg["has_annotation_sam"]),
    }
    return cfg, path_cfg


def resolve_path(root_dir, template, **kwargs):
    return os.path.join(root_dir, template.format(**kwargs))


def extract_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frames = []
    success, frame = video.read()
    while success:
        frames.append(frame)
        success, frame = video.read()
    video.release()
    return np.array(frames)


def load_new_config(path):
    return pickle.loads(np.load(path)['arr_0'])


def check_person(model_sam, path_cfg, user_name, time, device="cuda:0"):
    user_history_path = os.path.join(path_cfg["user_sam_dir"], user_name + ".txt")
    if time > 0:
        user_history_path = user_history_path.replace(".txt", f"_{time}.txt")

    if not os.path.exists(user_history_path):
        print(f"User {user_name} has no annotation at time {time}")
        exit(1)

    with open(user_history_path) as f:
        user_ann_list = f.readlines()
    user_ann_list = [line.strip() for line in user_ann_list]
    for line in tqdm(user_ann_list, desc=f"Parsing {user_name} with time {time}"):
        parse_and_save_results(
            line, model_sam, path_cfg, user_name, time,
            device=device, skip=False
        )


def check_item(video_path, model_sam, path_cfg, user, time):
    parse_and_save_results(video_path, model_sam, path_cfg, user, time)


def video_path_to_unique_name(video_path):
    """Extract the filename (without extension) from a video path.
    e.g. 'xtrainer_test/bowls/episode_00000__grasp-the-bottle/episode_00000__grasp-the-bottle_cam_high.mp4'
      -> 'episode_00000__grasp-the-bottle_cam_high'
    """
    name = video_path.strip()
    # Take only the filename
    name = os.path.basename(name)
    # Remove .mp4 extension
    if name.endswith('.mp4'):
        name = name[:-4]
    return name

def parse_and_save_results(
    line, model_sam, path_cfg, user, time, device="cuda:0", skip=False
):
    root_dir = path_cfg["root_dir"]
    error_log = path_cfg["error_log"]
    video_name_no_ext = video_path_to_unique_name(line)

    video_save_path = resolve_path(root_dir, path_cfg["sam_video_save_tpl"], time=time, video_name=video_name_no_ext)
    sam_save_path = resolve_path(root_dir, path_cfg["sam_mask_save_tpl"], time=time, video_name=video_name_no_ext)
    model_config_path = resolve_path(root_dir, path_cfg["save_path_sam_tpl"], time=time, video_name=video_name_no_ext)

    # Skip if next annotation round already exists
    # sam_next_time_path = resolve_path(root_dir, path_cfg["save_path_sam_tpl"], time=time + 1, video_name=video_name_no_ext)
    # if os.path.exists(sam_next_time_path):
    #     return

    # Skip if already processed
    # if os.path.exists(video_save_path) and os.path.exists(sam_save_path):
    #     UPDATE_VIDEO_LIST.append(video_save_path)
    #     return

    try:
        model_config = load_new_config(model_config_path)
    except Exception as e:
        print(f"Error in {model_config_path} because {e}", flush=True)
        with open(error_log, "a") as f:
            f.write("\n" + model_config_path + "\t" + user + "\t" + str(e) + line)
        return

    if "video_path" not in model_config:
        print(f"Error in {model_config_path}", flush=True)
        with open(error_log, "a") as f:
            f.write("\n" + model_config_path + "\t" + user + "\t" + "no video path")
        return

    if model_config["is_finished"]:
        print(model_config["video_path"] + " is finished", flush=True)
        if time >= 1:
            FINISHED_VIDEO_LIST.append(model_config["video_path"])
        return

    if model_config["is_hard_sample"]:
        if model_config["hard_sample_type"] == "困难样本":
            print(model_config["video_path"] + " is hard sample", flush=True)
            if time >= 1:
                HARD_SAMPLE_LIST.append(model_config["video_path"])
            return
        elif model_config["hard_sample_type"] == "问题样本":
            print(model_config["video_path"] + " is question sample", flush=True)
            if time >= 1:
                QUESTION_SAMPLE_LIST.append(model_config["video_path"])
            return

    # Build absolute video path from video_dir + relative video_path
    abs_video_path = os.path.join(path_cfg["video_dir"], model_config["video_path"])
    model_config["origin_video_path"] = abs_video_path
    model_config["video_path"] = abs_video_path

    try:
        mask_list = run_sam3_segmentation(
            model_sam, abs_video_path, model_config, device
        )
        os.makedirs(os.path.dirname(sam_save_path), exist_ok=True)
        np.savez_compressed(sam_save_path, masks=mask_list)

    except Exception as e:
        video_path = model_config["video_path"]
        print(f"Error in {video_path} because {e}", flush=True)
        with open(error_log, "a") as f:
            f.write("\n" + model_config["video_path"] + "\t" + user + "\t" + str(e))
        return

    origin_video_path = model_config["origin_video_path"]
    video = extract_frames(origin_video_path)
    if video.ndim != 4 or video.shape[0] == 0:
        print(f"Warning: failed to read video from {origin_video_path}, skipping visualization.", flush=True)
        UPDATE_VIDEO_LIST.append(video_save_path)
        return
    try:
        save_visualization_video(video, mask_list, video_save_path)
    except Exception as e:
        print(f"Warning: failed to synthesize video for {origin_video_path}: {e}", flush=True)
        UPDATE_VIDEO_LIST.append(video_save_path)
        return

    # Verify the written file
    file_size = os.path.getsize(video_save_path) if os.path.exists(video_save_path) else 0
    if file_size < 1024:
        print(f"Warning: written video file is suspiciously small ({file_size} bytes): {video_save_path}", flush=True)

    UPDATE_VIDEO_LIST.append(video_save_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--time", type=int, default=0)
    parser.add_argument("--sam3-ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()

    cfg, path_cfg = load_path_config(args.config)
    os.makedirs(os.path.dirname(path_cfg['sam_mask_save_tpl'].format(time=args.time, video_name='tmp')), exist_ok=True)
    os.makedirs(os.path.dirname(path_cfg['sam_video_save_tpl'].format(time=args.time, video_name='tmp')), exist_ok=True)

    sam_config = cfg["sam"]
    device = sam_config.get("device", "cuda:0")
    checkpoint_path = args.sam3_ckpt or sam_config.get("checkpoint")
    model_sam = build_sam3_model(
        device=device, checkpoint_path=checkpoint_path
    )
    check_person(
        model_sam, path_cfg, args.username, args.time,
        device=device
    )

    # === Post-processing: update annotation pool JSONs ===
    root_dir = path_cfg["root_dir"]

    # 1. Add processed videos to no_annotation_sam (queue for next round)
    add_file_number = 0
    with open(path_cfg["no_annotation_sam"], "r+") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        no_annotation = json.load(f)
        for video_path in UPDATE_VIDEO_LIST:
            add_file_number += 1
            video_name_no_ext = video_path_to_unique_name(video_path)
            save_path = resolve_path(
                root_dir, path_cfg["save_path_sam_tpl"], time=args.time + 1, video_name=video_name_no_ext
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
            video_name_no_ext = video_path_to_unique_name(video_path)
            if args.time == 0:
                # Previous source was the original video
                new_video_path = os.path.join(path_cfg["video_dir"], os.path.basename(video_path))
            else:
                # Previous source was sam_video from time-1
                new_video_path = resolve_path(
                    root_dir, path_cfg["sam_video_save_tpl"], time=args.time - 1, video_name=video_name_no_ext
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
