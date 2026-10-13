#!/usr/bin/env python3
"""HDF5 dataset loader for converted XTrainer episodes with 3-camera T-shape layout.

Three independent cameras (front + left_wrist + right_wrist) are concatenated into a
single T-shape image (front on top, left_wrist bottom-left, right_wrist bottom-right).

T-shape layout:
    ┌──────────────────────┐
    │    Front Camera       │  original size (H, W)
    │      (480 x 640)      │
    ├───────────┬──────────┤
    │ Left Wrist│Right Wrist│  each half size (H/2, W/2)
    │ (240x320) │(240x320)  │  (both are real, NOT copies)
    └───────────┴──────────┘
    Final: (H*1.5, W) = (720, 640)

Mask IO switches (aligned with model-side MaskWanMotConfig):
    - input_mask=True:  history + current masks read from npz (front input mask active).
                        In the current T-shape current frame, wrist regions are blacked out
                        (condition side) as before; history mask keeps real wrist masks.
    - input_mask=False: history mask + current mask → all zeros (skip npz read on input side).
    - predict_mask=True:  future masks read from npz (3 views, missing view → zeros).
    - predict_mask=False: future masks → all zeros.

Missing per-view npz never fails the episode (the view gets zeros). Missing mp4 still
fails the episode (mp4 is required for every camera).
"""

import json
import logging
import os
import pickle
import random
from pathlib import Path

import cv2
import decord
import h5py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

logger = logging.getLogger(__name__)


def get_history_indices(step_id, history_size, interval=1, random_sample=True):
    indices = []
    for i in range(history_size):
        end = step_id - (history_size - 1 - i) * interval
        if random_sample:
            start = max(end - interval + 1, 0)
            end = max(end, 0)
            idx = random.randint(start, end)
        else:
            idx = max(end, 0)
        indices.append(idx)
    indices[-1] = step_id
    return indices[:-1]


def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    padded = np.zeros(shape, dtype=vector.dtype)
    padded[..., : vector.shape[-1]] = vector
    return padded


# decord is FFmpeg-backed and 3–5× faster than cv2.VideoCapture for sparse
# random reads (cv2's per-frame POS_FRAMES seek scans up to GOP_LEN frames).
decord.bridge.set_bridge("native")


# ── T-shape concatenation helpers (3-camera version) ───────────────────────────

def concat_t_shape_image(
    front_img: np.ndarray, left_img: np.ndarray, right_img: np.ndarray
) -> np.ndarray:
    """Concatenate front + left_wrist + right_wrist images into T-shape layout.

    Args:
        front_img: (H, W, 3) front camera image.
        left_img:  (H, W, 3) left wrist camera image.
        right_img: (H, W, 3) right wrist camera image.

    Returns:
        T-shape concatenated image of shape (H + H//2, W, 3).
    """
    H, W = front_img.shape[:2]
    half_h, half_w = H // 2, W // 2

    left_resized = cv2.resize(left_img, (half_w, half_h))
    right_resized = cv2.resize(right_img, (half_w, half_h))

    bottom_row = np.hstack([left_resized, right_resized])
    combined = np.vstack([front_img, bottom_row])
    return combined


def concat_t_shape_images_batch(
    front_imgs: np.ndarray, left_imgs: np.ndarray, right_imgs: np.ndarray
) -> np.ndarray:
    """Batch version: (N, H, W, 3) front/left/right → T-shape."""
    return np.stack([
        concat_t_shape_image(front_imgs[i], left_imgs[i], right_imgs[i])
        for i in range(front_imgs.shape[0])
    ])


def concat_t_shape_mask(
    front_mask: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    wrist_blackout: bool = False,
) -> np.ndarray:
    """Concatenate front + left_wrist + right_wrist masks into T-shape layout.

    Args:
        front_mask: (H, W) front camera mask.
        left_mask:  (H, W) left wrist camera mask.
        right_mask: (H, W) right wrist camera mask.
        wrist_blackout: If True, zero out both wrist regions (used for current_mask).

    Returns:
        T-shape concatenated mask of shape (H + H//2, W).
    """
    H, W = front_mask.shape[:2]
    half_h, half_w = H // 2, W // 2

    if wrist_blackout:
        left_resized = np.zeros((half_h, half_w), dtype=front_mask.dtype)
        right_resized = np.zeros((half_h, half_w), dtype=front_mask.dtype)
    else:
        left_resized = cv2.resize(left_mask, (half_w, half_h), interpolation=cv2.INTER_NEAREST)
        right_resized = cv2.resize(right_mask, (half_w, half_h), interpolation=cv2.INTER_NEAREST)

    bottom_row = np.hstack([left_resized, right_resized])
    combined = np.vstack([front_mask, bottom_row])
    return combined


def concat_t_shape_masks_batch(
    front_masks: np.ndarray,
    left_masks: np.ndarray,
    right_masks: np.ndarray,
    wrist_blackout: bool = False,
) -> np.ndarray:
    """Batch version: (N, H, W) front/left/right masks → T-shape."""
    return np.stack([
        concat_t_shape_mask(
            front_masks[i], left_masks[i], right_masks[i], wrist_blackout=wrist_blackout
        )
        for i in range(front_masks.shape[0])
    ])


# ── Dataset class ──────────────────────────────────────────────────────────────

class HDF5VLADataset:
    """T-shape multi-camera dataset for XTrainer episodes.

    Always requires 2 cameras (front + left_wrist). Concatenates them into
    T-shape layout before returning samples.
    """

    def __init__(self, cfg, sample_weights=dict) -> None:
        self.cfg = cfg
        self.ACTION_TYPE = cfg.dataset.action_type

        self.hdf5_root = cfg.dataset.hdf5_dir
        self.DATASET_NAMES = list(sample_weights.keys())
        self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        self.STATE_DIM = cfg.dataset.state_dim
        self.min_episode_length = getattr(cfg.dataset, "min_episode_length", 1)
        self.segmentation_mode = getattr(cfg.dataset, "segmentation_mode", "union")

        # 3-camera T-shape: front + left_wrist + right_wrist (all real, independent)
        self.num_cameras = 3

        # Mask IO switches (aligned with MaskWanMotConfig.input_mask / predict_mask).
        # NOTE: cfg is an OmegaConf DictConfig; `getattr(cfg.training, key, default)` does NOT
        # return the default for MISSING keys (OmegaConf raises instead of falling through),
        # so we use explicit `in` checks.
        _tcfg = cfg.training if hasattr(cfg, "training") else None
        _use_mask_legacy = bool(_tcfg.use_mask) if (_tcfg is not None and "use_mask" in _tcfg) else True
        self.input_mask = (
            bool(_tcfg.input_mask) if (_tcfg is not None and "input_mask" in _tcfg) else _use_mask_legacy
        )
        self.predict_mask = (
            bool(_tcfg.predict_mask) if (_tcfg is not None and "predict_mask" in _tcfg) else False
        )
        print(
            f"[HDF5VLADataset_xtrainer_T] input_mask={self.input_mask}, "
            f"predict_mask={self.predict_mask}, use_mask(legacy)={_use_mask_legacy}"
        )

        self.deterministic_index_sampling = getattr(cfg.dataset, "deterministic_index_sampling", False)

        # Consistent center crop (applied to front camera BEFORE T-shape concat)
        self.consistent_crop = getattr(cfg.dataset, "consistent_crop", False)

        # Video generation parameters
        self.num_video_frames = cfg.dataset.num_video_frames
        self.video_action_freq_ratio = 6
        self.global_downsample_rate = 1
        self.CHUNK_SIZE = self.num_video_frames * self.video_action_freq_ratio

        # State noise augmentation (RDT-style, training only)
        self.state_noise_snr = getattr(cfg.dataset, "state_noise_snr", None)
        if self.state_noise_snr is not None:
            print(f"  [StateNoise] Enabled: SNR={self.state_noise_snr} dB")

        # Normalization
        self.use_self_norm = False
        mean_std_path = getattr(cfg.dataset, "mean_std_path", None)
        if mean_std_path is not None:
            if os.path.exists(mean_std_path):
                self.use_self_norm = True
                state_key = f"{self.ACTION_TYPE}_mean"
                state_std_key = f"{self.ACTION_TYPE}_std"
                with open(mean_std_path, "rb") as f:
                    norm_info = pickle.load(f)
                    self.qpos_mean = np.array(norm_info[state_key], dtype=np.float32)
                    self.qpos_std = np.array(norm_info[state_std_key], dtype=np.float32)
                    self.act_mean = np.array(norm_info["action_mean"], dtype=np.float32)
                    self.act_std = np.array(norm_info["action_std"], dtype=np.float32)
                    print(f"===Load mean_std from {mean_std_path}")
            else:
                raise ValueError(f"File does not exist: {mean_std_path}")

        # Episode file paths and metadata
        self.file_paths_dict = {}
        for name in self.DATASET_NAMES:
            self.file_paths_dict[name] = []

        dataset_json_path = Path(cfg.dev_dir) / "configs" / "dataset_meta" / "hdf5_robointer_T_list.json"

        print("Scanning T-shape dataset episodes ==========================")
        num_total = 0
        num_valid = 0
        num_timesteps = 0

        for dataset_name in self.DATASET_NAMES:
            print(f"Scanning {dataset_name}...")
            hdf5_root = Path(self.hdf5_root)
            dataset_root = (
                hdf5_root
                if hdf5_root.name == dataset_name
                else hdf5_root / dataset_name
            )
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

            episode_paths = sorted(dataset_root.rglob("episode.hdf5"))
            valid_paths = []
            episode_lens = []

            for hdf5_path in episode_paths:
                ep_dir = hdf5_path.parent
                # 3-camera T-shape: all three mp4 are required; npz are optional
                # (missing per-view npz is tolerated at sample time via zeros fill-in).
                required = [
                    "cam_high.mp4",
                    "cam_left_wrist.mp4",
                    "cam_right_wrist.mp4",
                    "instructions.json",
                    "meta.json",
                ]
                if not all((ep_dir / r).exists() for r in required):
                    continue

                valid, res = self.parse_hdf5_file_state_only(str(hdf5_path))
                _len = res["state"].shape[0] if valid else 0
                if _len < self.min_episode_length:
                    continue

                valid_paths.append(str(hdf5_path))
                episode_lens.append(_len)
                num_total += 1
                num_timesteps += _len
                if _len > 0:
                    num_valid += 1

            self.file_paths_dict[dataset_name] = valid_paths
            self.file_paths_dict[f"{dataset_name}_episode_weights"] = (
                np.array(episode_lens) / max(np.sum(episode_lens), 1)
            ).tolist()
            self.file_paths_dict[f"{dataset_name}_episode_timesteps"] = episode_lens
            self.file_paths_dict[f"{dataset_name}_total_timesteps"] = int(np.sum(episode_lens))

        self.file_paths_dict["total_hdf5"] = num_total
        self.file_paths_dict["total_valid_hdf5"] = num_valid
        self.file_paths_dict["total_valid_timesteps"] = num_timesteps

        os.makedirs(os.path.dirname(dataset_json_path), exist_ok=True)
        with open(dataset_json_path, "w", encoding="utf-8") as f:
            json.dump(self.file_paths_dict, f, ensure_ascii=False, indent=4)

        print(f"total_hdf5: {num_total}")
        print(f"valid_hdf5: {num_valid}")
        print(f"valid_timesteps: {num_timesteps}")

        # Dataset sampling weights
        dataset_weights = np.array([sample_weights[n] for n in self.DATASET_NAMES])
        self.dataset_weights = dataset_weights / dataset_weights.sum()
        for i, name in enumerate(self.DATASET_NAMES):
            print(f"+ Dataset {name} weight: {self.dataset_weights[i]}")

    def __len__(self):
        return sum(len(self.file_paths_dict[n]) for n in self.DATASET_NAMES)

    def get_dataset_name(self):
        return self.DATASET_NAMES

    def __getitem__(self, index):
        if self.deterministic_index_sampling:
            return self.get_item(index=index)
        return self.get_item()

    def get_item(self, index=None, state_only=False):
        while True:
            if index is None:
                dataset_name = np.random.choice(self.DATASET_NAMES, p=self.dataset_weights)
                file_path = np.random.choice(
                    self.file_paths_dict[dataset_name],
                    p=self.file_paths_dict[f"{dataset_name}_episode_weights"],
                )
            else:
                offset = 0
                for name in self.DATASET_NAMES:
                    paths = self.file_paths_dict[name]
                    if index < offset + len(paths):
                        dataset_name = name
                        file_path = paths[index - offset]
                        break
                    offset += len(paths)

            if state_only:
                valid, sample = self.parse_hdf5_file_state_only(file_path)
            else:
                valid, sample = self.parse_hdf5_file(file_path, dataset_name)
            if valid:
                return sample

    def load_state_action(self, file_path):
        with h5py.File(file_path, "r") as f:
            state = f["observations"]["state_qpos"][:].astype(np.float32)
            action = f["actions_eep"][:].astype(np.float32)
            return state, action

    def parse_hdf5_file_state_only(self, file_path):
        state, action = self.load_state_action(file_path)
        if len(state) < self.min_episode_length:
            return False, None
        return True, {
            "state": pad_vector(state, self.STATE_DIM),
            "action": pad_vector(action, self.STATE_DIM),
        }

    def _load_t5_embedding_for_instruction(self, episode_dir: Path, instruction_idx: int):
        t5_path = episode_dir / "t5_seen.pt"
        if not t5_path.exists():
            return None

        try:
            embedding_data = torch.load(t5_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            logger.warning(f"Failed to load T5 embedding from {t5_path}: {exc}")
            return None

        embedding = None
        if isinstance(embedding_data, list):
            if instruction_idx >= len(embedding_data):
                logger.warning(
                    f"Instruction index {instruction_idx} out of range for {t5_path} with {len(embedding_data)} embeddings"
                )
                return None
            embedding = embedding_data[instruction_idx]
        elif torch.is_tensor(embedding_data):
            if embedding_data.ndim == 3:
                if instruction_idx >= embedding_data.shape[0]:
                    return None
                embedding = embedding_data[instruction_idx]
            elif embedding_data.ndim == 2:
                embedding = embedding_data
            else:
                return None
        else:
            return None

        if not torch.is_tensor(embedding):
            embedding = torch.as_tensor(embedding)
        if embedding.ndim == 3 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim != 2:
            return None
        return embedding.float()

    # ── Crop helpers ──

    @staticmethod
    def _compute_center_crop(H: int, W: int, crop_h: int, crop_w: int):
        """Compute a center crop box of size (crop_h, crop_w).

        Returns:
            (y1, y2, x1, x2) crop coordinates, or None if image already <= crop size.
        """
        if H <= crop_h and W <= crop_w:
            return None
        crop_h = min(crop_h, H)
        crop_w = min(crop_w, W)
        y1 = max(0, (H - crop_h) // 2)
        x1 = max(0, (W - crop_w) // 2)
        return y1, y1 + crop_h, x1, x1 + crop_w

    @staticmethod
    def _apply_crop_images(images: np.ndarray, crop_box):
        """Crop (N, H, W, C) image array."""
        y1, y2, x1, x2 = crop_box
        return images[:, y1:y2, x1:x2, :]

    @staticmethod
    def _apply_crop_masks(masks: np.ndarray, crop_box):
        """Crop (N, H, W) mask array."""
        y1, y2, x1, x2 = crop_box
        return masks[:, y1:y2, x1:x2]

    # ── Video / segmentation parsing ──

    def parse_video(self, video_path, frame_indices, episode_length):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        frames = []
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count < episode_length:
                return []
            for fid in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                ok, frame = cap.read()
                if not ok or frame is None:
                    return []
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()

        imgs = np.stack(frames)
        if imgs.shape[0] < self.IMG_HISORY_SIZE:
            imgs = np.concatenate([
                np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                imgs
            ], axis=0)
        return imgs

    def parse_video_multi(self, video_path, index_groups, episode_length):
        """Read multiple groups of frame indices from a single mp4 with one open.

        Uses decord's ``get_batch`` for index-aware batched read (RGB output,
        ``num_threads=1`` to avoid CPU over-subscription across workers).

        Returns:
            list[np.ndarray] aligned with ``index_groups``, each
            (len(group), H, W, 3); padded to IMG_HISORY_SIZE on the time axis
            if shorter. Returns [] on file-level failure.
        """
        unique_indices = sorted({int(i) for g in index_groups for i in g})
        if not unique_indices:
            return [np.empty((0,), dtype=np.uint8) for _ in index_groups]

        try:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            n = len(vr)
            if n < episode_length or unique_indices[-1] >= n:
                return []
            frames = vr.get_batch(unique_indices).asnumpy()
        except Exception:
            return []
        frame_dict = {fid: frames[i] for i, fid in enumerate(unique_indices)}

        results = []
        for group in index_groups:
            if not group:
                results.append(np.empty((0,), dtype=np.uint8))
                continue
            imgs = np.stack([frame_dict[int(i)] for i in group])
            if imgs.shape[0] < self.IMG_HISORY_SIZE:
                imgs = np.concatenate([
                    np.tile(imgs[:1], (self.IMG_HISORY_SIZE - imgs.shape[0], 1, 1, 1)),
                    imgs
                ], axis=0)
            results.append(imgs)
        return results

    def parse_segmentation(self, seg_path, frame_indices, episode_length):
        with np.load(seg_path) as payload:
            if self.segmentation_mode == "instance":
                masks = payload["instance_masks"].astype(np.float32)
                if masks.ndim != 4:
                    return []
                mask_seq = masks.max(axis=0)
            else:
                mask_seq = payload["union_mask"].astype(np.float32)

        if mask_seq.shape[0] < episode_length:
            return []
        return np.stack([mask_seq[int(i)] for i in frame_indices])

    def _safe_parse_segmentation_multi(
        self, seg_path, index_groups, episode_length, shape_hw, view_tag: str
    ):
        """Like ``parse_segmentation_multi`` but never fails: any file-level or
        per-group failure is replaced by zero-filled arrays of ``shape_hw`` (=(H, W))
        with the expected group length. Missing file also yields zeros.

        Use this for mask npz reads that are allowed to be missing per view.
        """
        H, W = shape_hw

        def zero_groups():
            return [
                np.zeros((len(g), H, W), dtype=np.float32) if len(g) > 0 else np.empty((0,), dtype=np.float32)
                for g in index_groups
            ]

        if not os.path.exists(seg_path):
            return zero_groups()

        try:
            groups = self.parse_segmentation_multi(seg_path, index_groups, episode_length)
        except Exception as exc:  # pragma: no cover — logged and fall through
            logger.warning(f"[{view_tag}] Failed to read {seg_path}: {exc}; falling back to zeros.")
            return zero_groups()

        if not groups:
            return zero_groups()

        fixed = []
        for g_idx, arr in enumerate(groups):
            expected = len(index_groups[g_idx])
            if expected == 0:
                fixed.append(np.empty((0,), dtype=np.float32))
                continue
            if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[0] != expected:
                fixed.append(np.zeros((expected, H, W), dtype=np.float32))
            else:
                fixed.append(arr.astype(np.float32))
        return fixed

    def parse_segmentation_multi(self, seg_path, index_groups, episode_length):
        """Index multiple groups of frames from one ``np.load`` of an npz file.

        ``np.load`` on an npz triggers a fresh zip directory parse + memmap;
        loading + .astype(float32) of ``union_mask`` is the dominant cost when
        called repeatedly for the same file.

        Args:
            seg_path: path to .npz file.
            index_groups: list of lists of frame indices.
            episode_length: expected episode length for validation.

        Returns:
            list[np.ndarray] aligned with ``index_groups``, each (len(group), H, W).
            Returns [] on file-level failure.
        """
        with np.load(seg_path) as payload:
            if self.segmentation_mode == "instance":
                masks = payload["instance_masks"].astype(np.float32)
                if masks.ndim != 4:
                    return []
                mask_seq = masks.max(axis=0)
            else:
                mask_seq = payload["union_mask"].astype(np.float32)

        if mask_seq.shape[0] < episode_length:
            return []

        results = []
        for group in index_groups:
            if not group:
                results.append(np.empty((0,), dtype=np.float32))
                continue
            results.append(np.stack([mask_seq[int(i)] for i in group]))
        return results

    # ── Main parse function ──

    def parse_hdf5_file(self, file_path, dataset_name):
        state_seq, action_seq = self.load_state_action(file_path)
        num_steps = state_seq.shape[0]
        if num_steps < self.min_episode_length:
            return False, None

        episode_dir = Path(file_path).parent

        # Load instruction
        with open(episode_dir / "instructions.json", "r") as f:
            instr_dict = json.load(f)
        instructions = instr_dict.get("seen", [])
        if not instructions:
            return False, None
        instruction_idx = random.randint(0, len(instructions) - 1)
        instruction = instructions[instruction_idx]
        t5_embed = self._load_t5_embedding_for_instruction(episode_dir, instruction_idx)

        # Load meta
        with open(episode_dir / "meta.json", "r") as f:
            meta_json = json.load(f)

        # Skip idle/still steps
        EPS = 1e-2
        state_delta = np.abs(state_seq - state_seq[0:1])
        moving_indices = np.where(np.any(state_delta > EPS, axis=1))[0]
        if len(moving_indices) > 0:
            first_moving_idx = max(int(moving_indices[0]) - 1, 0)
        else:
            return False, None

        # Sample timestep
        step_id = random.randint(first_moving_idx, num_steps - 1)

        # History indices
        indices = get_history_indices(step_id, self.IMG_HISORY_SIZE)
        image_frame_indices = indices + [step_id]

        # Future frame indices
        future_indices = []
        for i in range(self.num_video_frames):
            idx = step_id + (i + 1) * self.video_action_freq_ratio * self.global_downsample_rate
            future_indices.append(min(idx, num_steps - 1))

        # Actions
        actions_history = action_seq[indices] if indices else np.empty((0, action_seq.shape[1]), dtype=np.float32)
        actions_future = action_seq[step_id : step_id + self.CHUNK_SIZE * self.global_downsample_rate : self.global_downsample_rate]
        if len(actions_future) == 0:
            return False, None
        actions = np.concatenate([actions_history, actions_future], axis=0)
        if actions.shape[0] < self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1:
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE + self.IMG_HISORY_SIZE - 1 - actions.shape[0], 1))
            ], axis=0)

        # State
        state = state_seq[step_id:step_id + 1]
        state_std = np.std(state_seq, axis=0)
        state_mean = np.mean(state_seq, axis=0)
        state_norm = np.sqrt(np.mean(state_seq ** 2, axis=0))

        if self.use_self_norm:
            state = (state - self.qpos_mean) / self.qpos_std
            actions = (actions - self.act_mean) / self.act_std

        # RDT-style state noise augmentation (training only)
        if self.state_noise_snr is not None:
            snr_linear = 10.0 ** (self.state_noise_snr / 10.0)
            if self.use_self_norm:
                # Normalized state: σ_state ≈ 1.0 per dim
                noise_std = 1.0 / np.sqrt(snr_linear)
            else:
                # Raw state: scale by per-dim episode std
                noise_std = state_std / np.sqrt(snr_linear)
            state = state + np.random.normal(0.0, noise_std, size=state.shape).astype(np.float32)

        state = pad_vector(state, self.STATE_DIM)
        state_indicator = pad_vector(np.ones_like(state_std), self.STATE_DIM)
        state_std = pad_vector(state_std, self.STATE_DIM)
        state_mean = pad_vector(state_mean, self.STATE_DIM)
        state_norm = pad_vector(state_norm, self.STATE_DIM)
        actions = pad_vector(actions, self.STATE_DIM)

        # ── Load front camera frames (required mp4) ──
        front_video_groups = self.parse_video_multi(
            episode_dir / "cam_high.mp4",
            [image_frame_indices, future_indices],
            num_steps,
        )
        if not front_video_groups or len(front_video_groups[0]) == 0:
            return False, None
        cam_high, cam_high_future = front_video_groups
        if len(cam_high_future) == 0:
            cam_high_future = np.repeat(cam_high[-1:], len(future_indices), axis=0)

        # ── Load left wrist camera frames (required mp4) ──
        left_video_groups = self.parse_video_multi(
            episode_dir / "cam_left_wrist.mp4",
            [image_frame_indices, future_indices],
            num_steps,
        )
        if not left_video_groups or len(left_video_groups[0]) == 0:
            return False, None
        cam_left, cam_left_future = left_video_groups
        if len(cam_left_future) == 0:
            cam_left_future = np.repeat(cam_left[-1:], len(future_indices), axis=0)

        # ── Load right wrist camera frames (required mp4, independent from left) ──
        right_video_groups = self.parse_video_multi(
            episode_dir / "cam_right_wrist.mp4",
            [image_frame_indices, future_indices],
            num_steps,
        )
        if not right_video_groups or len(right_video_groups[0]) == 0:
            return False, None
        cam_right, cam_right_future = right_video_groups
        if len(cam_right_future) == 0:
            cam_right_future = np.repeat(cam_right[-1:], len(future_indices), axis=0)

        # ── Mask reads are gated by input_mask / predict_mask ──
        # For each view, decide which groups need real npz data and which can be zeros.
        #   input_mask  → history (T_hist frames) + first_moving (1 frame, front only)
        #   predict_mask → future (T_fut frames)
        #
        # When a flag is False, we pass an EMPTY index list for that group, so the
        # npz is not paid for (and downstream code fills zeros).
        front_h, front_w = cam_high.shape[1:3]
        left_h_raw, left_w_raw = cam_left.shape[1:3]
        right_h_raw, right_w_raw = cam_right.shape[1:3]

        T_hist = len(image_frame_indices)
        T_fut = len(future_indices)

        # Front npz (has first_moving group).
        front_groups_req = [
            image_frame_indices if self.input_mask else [],
            [first_moving_idx] if self.input_mask else [],
            future_indices if self.predict_mask else [],
        ]
        if self.input_mask or self.predict_mask:
            seg_mask, first_moving_mask, seg_mask_future = self._safe_parse_segmentation_multi(
                episode_dir / "segmentation_primary.npz",
                front_groups_req,
                num_steps,
                shape_hw=(front_h, front_w),
                view_tag="front",
            )
        else:
            seg_mask = np.zeros((T_hist, front_h, front_w), dtype=np.float32)
            first_moving_mask = np.zeros((1, front_h, front_w), dtype=np.float32)
            seg_mask_future = np.zeros((T_fut, front_h, front_w), dtype=np.float32)

        # Ensure the returned arrays have the exact expected lengths (fill zeros otherwise).
        if self.input_mask:
            if seg_mask.shape[0] != T_hist:
                seg_mask = np.zeros((T_hist, front_h, front_w), dtype=np.float32)
            if first_moving_mask.shape[0] != 1:
                first_moving_mask = np.zeros((1, front_h, front_w), dtype=np.float32)
            # Original behavior: overwrite current-frame mask with the first-moving mask
            # so the condition reflects the "goal" segment; only meaningful when we actually read it.
            if seg_mask.shape[1:3] != (front_h, front_w):
                seg_mask = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in seg_mask])
            if first_moving_mask.shape[1:3] != (front_h, front_w):
                first_moving_mask = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in first_moving_mask])
            seg_mask[-1] = first_moving_mask[0]
        else:
            seg_mask = np.zeros((T_hist, front_h, front_w), dtype=np.float32)

        if self.predict_mask:
            if seg_mask_future.shape[0] != T_fut:
                seg_mask_future = np.zeros((T_fut, front_h, front_w), dtype=np.float32)
            if seg_mask_future.shape[1:3] != (front_h, front_w):
                seg_mask_future = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in seg_mask_future])
        else:
            seg_mask_future = np.zeros((T_fut, front_h, front_w), dtype=np.float32)

        # Left wrist npz (no first_moving).
        if self.input_mask or self.predict_mask:
            left_seg_mask, left_seg_mask_future = self._safe_parse_segmentation_multi(
                episode_dir / "segmentation_primary_left_wrist.npz",
                [
                    image_frame_indices if self.input_mask else [],
                    future_indices if self.predict_mask else [],
                ],
                num_steps,
                shape_hw=(left_h_raw, left_w_raw),
                view_tag="left_wrist",
            )
        else:
            left_seg_mask = np.zeros((T_hist, left_h_raw, left_w_raw), dtype=np.float32)
            left_seg_mask_future = np.zeros((T_fut, left_h_raw, left_w_raw), dtype=np.float32)

        if self.input_mask:
            if left_seg_mask.shape[0] != T_hist:
                left_seg_mask = np.zeros((T_hist, left_h_raw, left_w_raw), dtype=np.float32)
            if left_seg_mask.shape[1:3] != (left_h_raw, left_w_raw):
                left_seg_mask = np.stack([cv2.resize(m, (left_w_raw, left_h_raw), interpolation=cv2.INTER_NEAREST) for m in left_seg_mask])
        else:
            left_seg_mask = np.zeros((T_hist, left_h_raw, left_w_raw), dtype=np.float32)

        if self.predict_mask:
            if left_seg_mask_future.shape[0] != T_fut:
                left_seg_mask_future = np.zeros((T_fut, left_h_raw, left_w_raw), dtype=np.float32)
            if left_seg_mask_future.shape[1:3] != (left_h_raw, left_w_raw):
                left_seg_mask_future = np.stack([cv2.resize(m, (left_w_raw, left_h_raw), interpolation=cv2.INTER_NEAREST) for m in left_seg_mask_future])
        else:
            left_seg_mask_future = np.zeros((T_fut, left_h_raw, left_w_raw), dtype=np.float32)

        # Right wrist npz (no first_moving).
        if self.input_mask or self.predict_mask:
            right_seg_mask, right_seg_mask_future = self._safe_parse_segmentation_multi(
                episode_dir / "segmentation_primary_right_wrist.npz",
                [
                    image_frame_indices if self.input_mask else [],
                    future_indices if self.predict_mask else [],
                ],
                num_steps,
                shape_hw=(right_h_raw, right_w_raw),
                view_tag="right_wrist",
            )
        else:
            right_seg_mask = np.zeros((T_hist, right_h_raw, right_w_raw), dtype=np.float32)
            right_seg_mask_future = np.zeros((T_fut, right_h_raw, right_w_raw), dtype=np.float32)

        if self.input_mask:
            if right_seg_mask.shape[0] != T_hist:
                right_seg_mask = np.zeros((T_hist, right_h_raw, right_w_raw), dtype=np.float32)
            if right_seg_mask.shape[1:3] != (right_h_raw, right_w_raw):
                right_seg_mask = np.stack([cv2.resize(m, (right_w_raw, right_h_raw), interpolation=cv2.INTER_NEAREST) for m in right_seg_mask])
        else:
            right_seg_mask = np.zeros((T_hist, right_h_raw, right_w_raw), dtype=np.float32)

        if self.predict_mask:
            if right_seg_mask_future.shape[0] != T_fut:
                right_seg_mask_future = np.zeros((T_fut, right_h_raw, right_w_raw), dtype=np.float32)
            if right_seg_mask_future.shape[1:3] != (right_h_raw, right_w_raw):
                right_seg_mask_future = np.stack([cv2.resize(m, (right_w_raw, right_h_raw), interpolation=cv2.INTER_NEAREST) for m in right_seg_mask_future])
        else:
            right_seg_mask_future = np.zeros((T_fut, right_h_raw, right_w_raw), dtype=np.float32)

        # ── Consistent center crop (front camera only, before T-shape concat) ──
        if self.consistent_crop:
            crop_box = self._compute_center_crop(cam_high.shape[1], cam_high.shape[2], 320, 432)
            if crop_box is not None:
                cam_high = self._apply_crop_images(cam_high, crop_box)
                cam_high_future = self._apply_crop_images(cam_high_future, crop_box)
                seg_mask = self._apply_crop_masks(seg_mask, crop_box)
                seg_mask_future = self._apply_crop_masks(seg_mask_future, crop_box)

        # ── Resize wrist views (images + masks) to match front spatial size ──
        front_h, front_w = cam_high.shape[1:3]
        if (left_h_raw, left_w_raw) != (front_h, front_w):
            cam_left = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_left])
            cam_left_future = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_left_future])
            left_seg_mask = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in left_seg_mask])
            left_seg_mask_future = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in left_seg_mask_future])
        if (right_h_raw, right_w_raw) != (front_h, front_w):
            cam_right = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_right])
            cam_right_future = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_right_future])
            right_seg_mask = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in right_seg_mask])
            right_seg_mask_future = np.stack([cv2.resize(m, (front_w, front_h), interpolation=cv2.INTER_NEAREST) for m in right_seg_mask_future])

        # ── T-shape concatenation (3-camera) ──
        t_cam_high = concat_t_shape_images_batch(cam_high, cam_left, cam_right)
        t_cam_high_future = concat_t_shape_images_batch(cam_high_future, cam_left_future, cam_right_future)

        # History mask sequence (input-side): zeros when input_mask=False.
        t_seg_mask = concat_t_shape_masks_batch(
            seg_mask, left_seg_mask, right_seg_mask, wrist_blackout=False
        )

        # Current condition mask: front real + wrist blacked out (kept for backward-compat).
        # When input_mask=False, seg_mask[-1] is already zeros → the whole thing becomes zeros.
        t_current_seg_mask = concat_t_shape_mask(
            seg_mask[-1], left_seg_mask[-1], right_seg_mask[-1], wrist_blackout=True
        ).astype(np.float32)

        # Future target mask: predict_mask=False → all zeros by construction.
        t_seg_mask_future = concat_t_shape_masks_batch(
            seg_mask_future, left_seg_mask_future, right_seg_mask_future, wrist_blackout=False
        )

        meta = {
            "dataset_name": dataset_name,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction,
            "instruction_idx": instruction_idx,
            "task": meta_json.get("task_text", ""),
        }

        result = {
            "meta": meta,
            "state": state,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "actions": actions,
            "state_indicator": state_indicator,
            # T-shape concatenated outputs (single-view interface)
            "cam_high": t_cam_high,
            "cam_high_det_mask": t_seg_mask.astype(np.float32),
            "current_image": t_cam_high[-1],
            "future_images": t_cam_high_future,
            "current_seg_mask": t_current_seg_mask,
            "future_seg_masks": t_seg_mask_future.astype(np.float32),
            "t5_embed": t5_embed,
        }
        return True, result


import hydra


@hydra.main(version_base=None, config_path="../configs", config_name="train_maskwam")
def test(cfg):
    import sys
    from hydra.utils import instantiate, get_original_cwd

    project_root = get_original_cwd()
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ.setdefault("DEV_PATH", os.path.dirname(project_root))

    for baseset_dict in cfg.dataset.dataset_list:
        baseset = instantiate(baseset_dict, cfg=cfg, _recursive_=False)
        print("datasets:", baseset.get_dataset_name())
        print("num_episodes:", len(baseset))
        sample = baseset.get_item()
        for k, v in sample.items():
            if hasattr(v, "shape"):
                print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    test()
