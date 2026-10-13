#!/usr/bin/env python3
"""Convert xtrainer HDF5 episodes into episode-folder datasets.

This script reads xtrainer-format HDF5 files (collected from real robot) and
converts them into episode folders:

    <output-dir>/
      episode_00000__<task-slug>/
        episode.hdf5
        cam_high.mp4
        cam_left_wrist.mp4
        cam_right_wrist.mp4
        instructions.json
        meta.json

By default all supported views are extracted. Multiple camera views are
stored as multiple MP4 files under the same episode directory.

The ``episode.hdf5`` stores:
  - observations/state_eep   (N, 16)  — from eepos (end-effector pose)
  - observations/state_qpos  (N, 14)  — from qpos  (joint positions)
  - actions_eep               (N, action_dim) — from action

Since xtrainer data has different dimensionality than DROID (14-dim action,
16-dim eepos, 14-dim qpos), the arrays are stored as-is without truncation.

Segmentation masks (``segmentation_primary.npz``) are not available for
xtrainer data and are therefore skipped.

Dependencies:
  - h5py
  - numpy
  - cv2

Typical usage:

    python convert_xtrainer_to_episode_folders.py \\
        --input-dir /path/to/xtrainer/bowls \\
        --output-dir converted_xtrainer/bowls

    python convert_xtrainer_to_episode_folders.py \\
        --input-dir /path/to/xtrainer/bowls \\
        --output-dir converted_xtrainer/bowls \\
        --camera-views cam_high cam_left_wrist cam_right_wrist
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FFMPEG_CANDIDATES = [
    "ffmpeg",
]

CAMERA_VIEWS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# State keys to read from xtrainer HDF5
STATE_KEY_EEP = "observations/eepos"
STATE_KEY_QPOS = "observations/qpos"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert xtrainer HDF5 episodes into episode-folder datasets."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing xtrainer .hdf5 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for converted episode folders.",
    )
    parser.add_argument(
        "--camera-views",
        nargs="+",
        default=CAMERA_VIEWS,
        help=(
            "Camera views to extract. "
            f"Default: {' '.join(CAMERA_VIEWS)}."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Output video FPS. Default: 30.0 (matching xtrainer source).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip episodes whose output folder already exists.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Stop after converting this many episodes.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        type=Path,
        default=None,
        help="Optional path to ffmpeg binary.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unknown-task"


def resolve_ffmpeg(requested_path: Optional[Path]) -> Optional[Path]:
    """Find a usable ffmpeg binary."""
    if requested_path is not None:
        resolved = requested_path.expanduser().resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"ffmpeg not found: {resolved}")

    for candidate in DEFAULT_FFMPEG_CANDIDATES:
        which_path = shutil.which(candidate)
        if which_path is not None:
            return Path(which_path)
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return candidate_path.resolve()
    return None


# ---------------------------------------------------------------------------
# HDF5 reading
# ---------------------------------------------------------------------------

def read_xtrainer_hdf5(
    path: Path,
    camera_views: Optional[Sequence[str]] = None,
) -> dict:
    """Read an xtrainer HDF5 file and return a structured dict.

    Returns
    -------
    dict with keys:
        instruction : str
        action      : np.ndarray (N, action_dim)
        state_eep   : np.ndarray | None (N, 16) — eepos
        state_qpos  : np.ndarray | None (N, 14) — qpos
        images      : dict[camera_view] -> list of np.ndarray (JPEG bytes)
        num_frames  : int
    """
    if camera_views is None:
        camera_views = CAMERA_VIEWS

    with h5py.File(path, "r") as f:
        # Action
        action = f["action"][:].astype(np.float32)
        num_frames = action.shape[0]

        # Instruction (same for all frames, take first)
        raw_instr = f["instruction"][0]
        if isinstance(raw_instr, bytes):
            instruction = raw_instr.decode("utf-8")
        else:
            instruction = str(raw_instr)

        # State: read both eepos and qpos if available
        state_eep = None
        state_qpos = None
        if STATE_KEY_EEP in f:
            state_eep = f[STATE_KEY_EEP][:].astype(np.float32)
        if STATE_KEY_QPOS in f:
            state_qpos = f[STATE_KEY_QPOS][:].astype(np.float32)
        if state_eep is None and state_qpos is None:
            raise KeyError(
                f"No state key found in {path}. "
                f"Tried: {STATE_KEY_EEP}, {STATE_KEY_QPOS}"
            )

        # Images: read JPEG-encoded bytes for requested camera views only
        images: Dict[str, list] = {}
        for cam_view in camera_views:
            img_key = f"observations/images/{cam_view}"
            if img_key in f:
                images[cam_view] = [f[img_key][i] for i in range(num_frames)]

    return {
        "instruction": instruction,
        "action": action,
        "state_eep": state_eep,
        "state_qpos": state_qpos,
        "images": images,
        "num_frames": num_frames,
    }


# ---------------------------------------------------------------------------
# Episode writing
# ---------------------------------------------------------------------------

def write_episode_hdf5(
    path: Path,
    state_eep: Optional[np.ndarray],
    state_qpos: Optional[np.ndarray],
    action: np.ndarray,
) -> None:
    """Write episode.hdf5 with both state_eep and state_qpos when available."""
    ensure_parent(path)
    with h5py.File(path, "w") as f:
        obs = f.create_group("observations")
        if state_eep is not None:
            obs.create_dataset("state_eep", data=state_eep, compression="gzip")
        if state_qpos is not None:
            obs.create_dataset("state_qpos", data=state_qpos, compression="gzip")
        f.create_dataset("actions_eep", data=action, compression="gzip")


def write_instructions_json(path: Path, instruction: str) -> None:
    """Write instructions.json with the 'seen' field."""
    write_json(path, {"seen": [instruction]})


def write_meta_json(
    path: Path,
    *,
    source_hdf5: str,
    episode_index: int,
    camera_view: str,
    camera_views: Sequence[str],
    instruction: str,
    num_frames: int,
    state_eep_dim: Optional[int],
    state_qpos_dim: Optional[int],
    action_dim: int,
    fps: float,
    video_meta: dict,
    video_meta_by_view: Dict[str, dict],
) -> None:
    """Write meta.json with relevant metadata."""
    meta = {
        "source_dataset": "xtrainer",
        "source_hdf5": source_hdf5,
        "episode_index": episode_index,
        "camera_view": camera_view,
        "camera_views": list(camera_views),
        "task_text": instruction,
        "instruction_source": "hdf5",
        "instruction_candidates": [instruction],
        "frame_range": {
            "start": 0,
            "end_inclusive": num_frames - 1,
            "num_frames": num_frames,
        },
        "frame_count_before_crop": num_frames,
        "frame_count_after_crop": num_frames,
        "state_eep_dim": state_eep_dim,
        "state_qpos_dim": state_qpos_dim,
        "action_dim": action_dim,
        "video_meta": video_meta,
        "video_meta_by_view": video_meta_by_view,
    }
    write_json(path, meta)


def encode_video_ffmpeg(
    ffmpeg_path: Path,
    frames: list,
    output_path: Path,
    fps: float,
) -> dict:
    """Encode JPEG-compressed frames into an MP4 video using ffmpeg.

    Uses a temporary directory of JPEG files as input to avoid pipe
    buffer issues with large frame sequences.

    Returns video metadata dict.
    """
    first_img = cv2.imdecode(
        np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = first_img.shape[:2]

    ensure_parent(output_path)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)

        for i, jpeg_bytes in enumerate(frames):
            frame_path = tmp_dir_path / f"frame_{i:06d}.jpg"
            frame_path.write_bytes(bytes(jpeg_bytes))

        tmp_output = tmp_dir_path / "output.mp4"

        command = [
            str(ffmpeg_path),
            "-y",
            "-framerate", f"{fps:.8f}",
            "-i", str(tmp_dir_path / "frame_%06d.jpg"),
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(tmp_output),
        ]
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed while encoding {output_path.name}.\n"
                f"command={' '.join(command)}\n"
                f"stderr={completed.stderr.strip()}"
            )

        # Use shutil.move instead of Path.replace to support cross-device moves
        shutil.move(str(tmp_output), str(output_path))

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": len(frames),
        "video_encoder": "libx264",
        "video_pixel_format": "yuv420p",
    }


def encode_video_opencv(
    frames: list,
    output_path: Path,
    fps: float,
) -> dict:
    """Encode JPEG-compressed frames into an MP4 video using OpenCV.

    Fallback when ffmpeg is not available.
    Returns video metadata dict.
    """
    first_img = cv2.imdecode(
        np.frombuffer(frames[0], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    height, width = first_img.shape[:2]

    ensure_parent(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    written = 0
    for jpeg_bytes in frames:
        img = cv2.imdecode(
            np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if img is None:
            raise RuntimeError("Failed to decode JPEG frame")
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        writer.write(img)
        written += 1

    writer.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": written,
        "video_encoder": "mp4v",
        "video_pixel_format": "bgr24",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    fps = args.fps
    camera_views = args.camera_views

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    ffmpeg_path = resolve_ffmpeg(args.ffmpeg_path)

    hdf5_files = sorted(input_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[setup] input_dir={input_dir} output_dir={output_dir} "
        f"fps={fps} camera_views={camera_views} "
        f"ffmpeg={ffmpeg_path or 'not-found (using opencv)'} "
        f"hdf5_files={len(hdf5_files)}",
        flush=True,
    )

    counters = Counter()
    exported_examples: List[str] = []

    for ep_idx, hdf5_path in enumerate(hdf5_files):
        if args.max_episodes is not None and counters["exported"] >= args.max_episodes:
            break

        print(f"[{ep_idx + 1}/{len(hdf5_files)}] Processing {hdf5_path.name} ...", flush=True)

        try:
            data = read_xtrainer_hdf5(
                hdf5_path, camera_views=camera_views
            )
        except Exception as exc:
            print(f"  [error] Failed to read {hdf5_path.name}: {exc}", flush=True)
            counters["error_read"] += 1
            continue

        instruction = data["instruction"]
        task_slug = slugify(instruction)
        episode_name = f"episode_{ep_idx:05d}__{task_slug}"

        episode_dir = output_dir / episode_name
        if args.skip_existing and episode_dir.exists():
            counters["skip_existing"] += 1
            continue

        episode_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write shared files once per episode
        write_episode_hdf5(
            path=episode_dir / "episode.hdf5",
            state_eep=data["state_eep"],
            state_qpos=data["state_qpos"],
            action=data["action"],
        )
        write_instructions_json(
            path=episode_dir / "instructions.json",
            instruction=instruction,
        )

        # 2. Encode one MP4 per available camera view in the same folder
        available_views = [view for view in camera_views if view in data["images"]]
        missing_views = [view for view in camera_views if view not in data["images"]]
        for missing_view in missing_views:
            print(f"  [skip] camera view '{missing_view}' not found in {hdf5_path.name}", flush=True)
            counters["skip_missing_camera"] += 1

        video_meta_by_view: Dict[str, dict] = {}
        for cam_view in available_views:
            frames = data["images"][cam_view]
            video_filename = f"{cam_view}.mp4"
            try:
                if ffmpeg_path is not None:
                    video_meta = encode_video_ffmpeg(
                        ffmpeg_path=ffmpeg_path,
                        frames=frames,
                        output_path=episode_dir / video_filename,
                        fps=fps,
                    )
                else:
                    video_meta = encode_video_opencv(
                        frames=frames,
                        output_path=episode_dir / video_filename,
                        fps=fps,
                    )
            except Exception as exc:
                print(
                    f"  [error] Failed to encode video for {cam_view}: {exc}",
                    flush=True,
                )
                counters["error_video"] += 1
                continue

            video_meta_by_view[cam_view] = video_meta
            counters["videos_exported"] += 1
            print(
                f"  [ok] {cam_view}: {data['num_frames']} frames, "
                f"video={video_meta['width']}x{video_meta['height']}",
                flush=True,
            )

        if not video_meta_by_view:
            print("  [error] No videos were successfully encoded for this episode", flush=True)
            counters["error_no_video"] += 1
            continue

        # Prefer cam_high as primary view for backward compatibility.
        primary_view = "cam_high" if "cam_high" in video_meta_by_view else next(iter(video_meta_by_view))

        # 3. Write one meta.json for the whole episode
        write_meta_json(
            path=episode_dir / "meta.json",
            source_hdf5=hdf5_path.name,
            episode_index=ep_idx,
            camera_view=primary_view,
            camera_views=list(video_meta_by_view.keys()),
            instruction=instruction,
            num_frames=data["num_frames"],
            state_eep_dim=data["state_eep"].shape[1] if data["state_eep"] is not None else None,
            state_qpos_dim=data["state_qpos"].shape[1] if data["state_qpos"] is not None else None,
            action_dim=data["action"].shape[1],
            fps=fps,
            video_meta=video_meta_by_view[primary_view],
            video_meta_by_view=video_meta_by_view,
        )

        counters["exported"] += 1
        if len(exported_examples) < 10:
            exported_examples.append(str(episode_dir))

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "fps": fps,
        "camera_views": camera_views,
        "ffmpeg_path": str(ffmpeg_path) if ffmpeg_path else None,
        "counters": dict(counters),
        "examples": exported_examples,
    }
    write_json(output_dir / "conversion_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
