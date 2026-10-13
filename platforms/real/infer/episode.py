"""Read one observation from a converted XTrainer episode."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EpisodeFrame:
    cam_high: np.ndarray
    cam_left_wrist: np.ndarray
    cam_right_wrist: np.ndarray
    state: np.ndarray
    prompt: str
    frame_index: int


def _read_video_frame(path: Path, frame_index: int) -> np.ndarray:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"缺少视频: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise IndexError(f"无法读取 {path.name} 的第 {frame_index} 帧")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _find_state(handle, frame_index: int) -> np.ndarray:
    candidates = (
        "observations/state_qpos",
        "observations/state_eep",
        "state_qpos",
        "state_eep",
        "robot_states",
    )
    for key in candidates:
        if key in handle:
            values = handle[key]
            if frame_index >= len(values):
                raise IndexError(f"state 只有 {len(values)} 帧，无法读取 {frame_index}")
            return np.asarray(values[frame_index], dtype=np.float32)
    raise KeyError(f"episode.hdf5 中未找到 state，尝试过: {', '.join(candidates)}")


def _read_prompt(episode_dir: Path) -> str:
    path = episode_dir / "instructions.json"
    if not path.is_file():
        return "put the selected object into the container"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("seen", "instruction", "instructions", "task"):
            value = payload.get(key)
            if isinstance(value, list) and value:
                return str(value[0])
            if isinstance(value, str):
                return value
    if isinstance(payload, list) and payload:
        return str(payload[0])
    return "put the selected object into the container"


def load_episode_frame(episode_dir: str | Path, frame_index: int = 0) -> EpisodeFrame:
    import h5py

    directory = Path(episode_dir).expanduser().resolve()
    if frame_index < 0:
        raise ValueError("frame_index 不能为负数")
    hdf5_path = directory / "episode.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"缺少 episode.hdf5: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as handle:
        state = _find_state(handle, frame_index)
    return EpisodeFrame(
        cam_high=_read_video_frame(directory / "cam_high.mp4", frame_index),
        cam_left_wrist=_read_video_frame(directory / "cam_left_wrist.mp4", frame_index),
        cam_right_wrist=_read_video_frame(directory / "cam_right_wrist.mp4", frame_index),
        state=state,
        prompt=_read_prompt(directory),
        frame_index=frame_index,
    )
