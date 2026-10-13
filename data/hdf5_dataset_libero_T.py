#!/usr/bin/env python3
"""HDF5 dataset loader for converted LIBERO episodes with T-shape multi-camera concatenation.

Combines LIBERO data format (observations/robot_states, actions) with T-shape
multi-camera layout (front + wrist cameras concatenated into a single image).

Expected layout (produced by convert_to_vam.py):
    <hdf5_dir>/
      <dataset_name>/           # e.g. libero_10_robot
        task_xxxxx__task-name/
          episode_N/
            episode.hdf5           # observations/robot_states (T,9), actions (T,7)
            cam_high.mp4           # front camera video
            cam_left_wrist.mp4     # left wrist camera video
            cam_high_mask.mp4      # front mask video (RGB: red=object, blue=robot, black=bg)
            cam_left_wrist_mask.mp4 # wrist mask video (RGB: red=object, blue=robot, black=bg)
            instructions.json      # {"seen": ["..."]}

T-shape layout:
    ┌──────────────────────┐
    │    Front Camera       │  original size (H, W)
    │      (480 x 640)      │
    ├───────────┬──────────┤
    │ Left Wrist│Right Wrist│  each half size (H/2, W/2)
    │ (240x320) │(240x320)  │  (right = copy of left)
    └───────────┴──────────┘
    Final: (H*1.5, W) = (720, 640)

Mask rules:
    - current_mask: front mask + BLACK wrist regions (zeros)
    - future_masks: front mask + real wrist masks (both sides = left wrist mask)
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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

# decord is FFmpeg-backed and 3–5× faster than cv2.VideoCapture for sparse
# random reads (cv2's per-frame POS_FRAMES seek scans up to GOP_LEN frames).
decord.bridge.set_bridge("native")

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


def concat_t_shape_image(front_img: np.ndarray, wrist_img: np.ndarray) -> np.ndarray:
    height, width = front_img.shape[:2]
    wrist_resized = cv2.resize(wrist_img, (width // 2, height // 2))
    bottom_row = np.hstack([wrist_resized, wrist_resized.copy()])
    return np.vstack([front_img, bottom_row])


def concat_t_shape_images_batch(
    front_imgs: np.ndarray, wrist_imgs: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            concat_t_shape_image(front_imgs[i], wrist_imgs[i])
            for i in range(front_imgs.shape[0])
        ]
    )


# ── Dataset class ──────────────────────────────────────────────────────────────

class HDF5VLADataset:
    """T-shape multi-camera dataset for LIBERO episodes.

    Uses LIBERO's HDF5 format (observations/robot_states + actions) and
    concatenates front + wrist cameras into T-shape layout.
    """

    def __init__(self, cfg, sample_weights=dict) -> None:
        self.cfg = cfg
        self.ACTION_TYPE = cfg.dataset.action_type  # "qpos" for LIBERO

        self.hdf5_root = cfg.dataset.hdf5_dir
        self.DATASET_NAMES = list(sample_weights.keys())
        self.IMG_HISORY_SIZE = cfg.dataset.img_history_size
        self.STATE_DIM = cfg.dataset.state_dim
        self.min_episode_length = getattr(cfg.dataset, "min_episode_length", 1)
        self.deterministic_index_sampling = getattr(cfg.dataset, "deterministic_index_sampling", False)

        # T-shape always requires 2 cameras
        self.num_cameras = 2

        # Mask IO switches (aligned with MaskWanMotConfig.input_mask / predict_mask).
        _tcfg = cfg.training if hasattr(cfg, "training") else None
        _use_mask_legacy = bool(_tcfg.use_mask) if (_tcfg is not None and "use_mask" in _tcfg) else True
        self.input_mask = (
            bool(_tcfg.input_mask) if (_tcfg is not None and "input_mask" in _tcfg) else _use_mask_legacy
        )
        self.predict_mask = (
            bool(_tcfg.predict_mask) if (_tcfg is not None and "predict_mask" in _tcfg) else _use_mask_legacy
        )
        print(
            f"[HDF5VLADataset_libero_T] input_mask={self.input_mask}, "
            f"predict_mask={self.predict_mask}, use_mask(legacy)={_use_mask_legacy}"
        )

        # Video generation parameters (LIBERO frequency ratio)
        self.num_video_frames = cfg.dataset.num_video_frames
        self.video_action_freq_ratio = 6  # Video:Action frequency ratio
        self.global_downsample_rate = 1
        self.CHUNK_SIZE = self.num_video_frames * self.video_action_freq_ratio

        # Data augmentation
        self.image_aug = getattr(cfg.dataset, "image_aug", False)
        self.image_aug_prob = getattr(cfg.dataset, "image_aug_prob", 0.5)
        self.auto_adjust_image_brightness = getattr(cfg.dataset, "auto_adjust_image_brightness", False)
        self.geo_aug = getattr(cfg.dataset, "geo_aug", False)
        self.geo_aug_crop_ratio = getattr(cfg.dataset, "geo_aug_crop_ratio", 0.85)
        self.geo_aug_rotation_degrees = getattr(cfg.dataset, "geo_aug_rotation_degrees", 5)
        if self.image_aug:
            print(
                f"  [DataAug] Enabled: prob={self.image_aug_prob}, "
                f"auto_brightness={self.auto_adjust_image_brightness}, "
                f"geo_aug={self.geo_aug}(crop={self.geo_aug_crop_ratio}, rot={self.geo_aug_rotation_degrees}°)"
            )

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

        print("Scanning LIBERO T-shape dataset episodes ==========================")
        num_total = 0
        num_valid = 0
        num_timesteps = 0

        for dataset_name in self.DATASET_NAMES:
            print(f"Scanning {dataset_name}...")
            dataset_root = Path(self.hdf5_root) / dataset_name
            if not dataset_root.exists():
                raise FileNotFoundError(f"Dataset folder does not exist: {dataset_root}")

            episode_paths = sorted(dataset_root.rglob("episode.hdf5"))
            valid_paths = []
            episode_lens = []

            for hdf5_path in episode_paths:
                ep_dir = hdf5_path.parent
                # T-shape requires both cameras; mask files only when use_mask=True
                required = [
                    "cam_high.mp4",
                    "instructions.json",
                    "cam_left_wrist.mp4",
                ]
                if self.input_mask or self.predict_mask:
                    required += [
                        "cam_high_mask.mp4",
                        "cam_left_wrist_mask.mp4",
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
        """Load state and action from LIBERO HDF5 format."""
        with h5py.File(file_path, "r") as f:
            state = f["observations"]["robot_states"][:].astype(np.float32)  # (T, 9)
            action = f["actions"][:].astype(np.float32)                     # (T, 7)
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
                    f"Instruction index {instruction_idx} out of range for {t5_path} "
                    f"with {len(embedding_data)} embeddings"
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

    # ── Video / segmentation parsing ──

    @staticmethod
    def _remove_robot_arm(mask_frame: np.ndarray) -> np.ndarray:
        """Black out blue robot-arm pixels while preserving object colors."""
        semantic = mask_frame.astype(np.int16)
        red = semantic[..., 0]
        green = semantic[..., 1]
        blue = semantic[..., 2]
        robot_pixels = (blue > 64) & (blue > red) & (blue > green)
        result = mask_frame.copy()
        result[robot_pixels] = 0
        return result

    def parse_mask_video_all(self, video_path, all_frame_indices, episode_length):
        frames = self.parse_video_all(video_path, all_frame_indices, episode_length)
        return {
            frame_id: self._remove_robot_arm(frame)
            for frame_id, frame in frames.items()
        }

    def parse_video_all(self, video_path, all_frame_indices, episode_length):
        """Read all needed frames from an mp4 video in a single open via decord.

        Returns:
            dict mapping frame_index → (H, W, 3) uint8 RGB array, or {} on failure.
        """
        sorted_indices = sorted({int(i) for i in all_frame_indices})
        if not sorted_indices:
            return {}
        try:
            vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
            n = len(vr)
            if n < episode_length or sorted_indices[-1] >= n:
                return {}
            frames = vr.get_batch(sorted_indices).asnumpy()
        except Exception:
            return {}
        return {fid: frames[i] for i, fid in enumerate(sorted_indices)}

    def _gather_frames(self, frame_dict, indices):
        """Gather frames from a dict by indices, return (N, H, W, 3) array or []."""
        frames = []
        for idx in indices:
            if idx not in frame_dict:
                return []
            frames.append(frame_dict[idx])
        return np.stack(frames) if frames else []

    # ── Data augmentation ──

    def _apply_augmentation(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        images_future: np.ndarray,
        masks_future: np.ndarray,
    ):
        """Apply per-view augmentation with temporally consistent params.

        RGB masks are transformed with nearest interpolation to preserve color labels.
        """
        if not self.image_aug:
            return images, masks, images_future, masks_future

        if random.random() > self.image_aug_prob:
            return images, masks, images_future, masks_future

        if self.auto_adjust_image_brightness:
            images = self._auto_brightness(images)
            images_future = self._auto_brightness(images_future)

        aug_type = random.choice(["null", "color"])
        images, images_future = self._apply_color_aug(images, images_future, aug_type)

        if self.geo_aug:
            images, masks, images_future, masks_future = self._apply_geo_aug_rgb_masks(
                images, masks, images_future, masks_future
            )

        return images, masks, images_future, masks_future

    @staticmethod
    def _auto_brightness(images: np.ndarray) -> np.ndarray:
        """Auto-adjust brightness for dark frames. images: (N, H, W, 3) uint8."""
        result = []
        for img in images:
            avg_brightness = img.mean() / 255.0
            if avg_brightness <= 0.15:
                pil_img = Image.fromarray(img)
                pil_img = transforms.ColorJitter(brightness=(1.75, 1.75))(pil_img)
                img = np.array(pil_img)
            result.append(img)
        return np.stack(result)

    @staticmethod
    def _apply_color_aug(
        images: np.ndarray,
        images_future: np.ndarray,
        aug_type: str,
    ):
        """Apply color augmentation to image frames only."""
        if aug_type in {"null", "none"}:
            return images, images_future

        n_hist = images.shape[0]
        all_images = np.concatenate([images, images_future], axis=0)

        brightness_factor = random.uniform(0.85, 1.15)
        contrast_factor = random.uniform(0.85, 1.15)
        saturation_factor = random.uniform(0.85, 1.15)
        hue_factor = random.uniform(-0.02, 0.02)

        jittered = []
        for img_np in all_images:
            pil_img = Image.fromarray(img_np)
            pil_img = TF.adjust_brightness(pil_img, brightness_factor)
            pil_img = TF.adjust_contrast(pil_img, contrast_factor)
            pil_img = TF.adjust_saturation(pil_img, saturation_factor)
            pil_img = TF.adjust_hue(pil_img, hue_factor)
            jittered.append(np.array(pil_img))
        all_images = np.stack(jittered)

        return all_images[:n_hist], all_images[n_hist:]

    def _apply_geo_aug_rgb_masks(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        images_future: np.ndarray,
        masks_future: np.ndarray,
    ):
        """Apply crop/resize/rotation jointly to RGB images and RGB masks."""
        n_hist = images.shape[0]
        height, width = images.shape[1:3]
        crop_ratio = float(self.geo_aug_crop_ratio)
        crop_h = max(1, min(height, int(round(height * crop_ratio))))
        crop_w = max(1, min(width, int(round(width * crop_ratio))))
        top, left, crop_h, crop_w = transforms.RandomCrop.get_params(
            torch.empty(1, height, width), output_size=(crop_h, crop_w)
        )
        angle = random.uniform(-self.geo_aug_rotation_degrees, self.geo_aug_rotation_degrees)

        all_images = np.concatenate([images, images_future], axis=0)
        all_masks = np.concatenate([masks, masks_future], axis=0)

        aug_images = []
        aug_masks = []
        for img_np, mask_np in zip(all_images, all_masks):
            pil_img = Image.fromarray(img_np)
            pil_img = TF.crop(pil_img, top, left, crop_h, crop_w)
            pil_img = TF.resize(pil_img, [height, width], interpolation=InterpolationMode.BILINEAR)
            pil_img = TF.rotate(pil_img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            aug_images.append(np.array(pil_img, dtype=np.uint8))

            pil_mask = Image.fromarray(mask_np)
            pil_mask = TF.crop(pil_mask, top, left, crop_h, crop_w)
            pil_mask = TF.resize(pil_mask, [height, width], interpolation=InterpolationMode.NEAREST)
            pil_mask = TF.rotate(pil_mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            aug_masks.append(np.array(pil_mask, dtype=np.uint8))

        aug_images = np.stack(aug_images)
        aug_masks = np.stack(aug_masks)
        return (
            aug_images[:n_hist], aug_masks[:n_hist],
            aug_images[n_hist:], aug_masks[n_hist:],
        )

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

        # Actions: history + downsampled future
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

        # ── Load front camera frames (single open) ──
        all_front_indices = image_frame_indices + future_indices
        front_frames = self.parse_video_all(episode_dir / "cam_high.mp4", all_front_indices, num_steps)
        if not front_frames:
            return False, None

        cam_high = self._gather_frames(front_frames, image_frame_indices)
        if len(cam_high) == 0:
            return False, None
        if cam_high.shape[0] < self.IMG_HISORY_SIZE:
            cam_high = np.concatenate([
                np.tile(cam_high[:1], (self.IMG_HISORY_SIZE - cam_high.shape[0], 1, 1, 1)),
                cam_high
            ], axis=0)

        cam_high_future = self._gather_frames(front_frames, future_indices)
        if len(cam_high_future) == 0:
            cam_high_future = np.repeat(cam_high[-1:], len(future_indices), axis=0)

        # ── Load wrist camera frames (single open) ──
        wrist_frames = self.parse_video_all(episode_dir / "cam_left_wrist.mp4", all_front_indices, num_steps)
        if not wrist_frames:
            return False, None

        cam_wrist = self._gather_frames(wrist_frames, image_frame_indices)
        if len(cam_wrist) == 0:
            return False, None
        if cam_wrist.shape[0] < self.IMG_HISORY_SIZE:
            cam_wrist = np.concatenate([
                np.tile(cam_wrist[:1], (self.IMG_HISORY_SIZE - cam_wrist.shape[0], 1, 1, 1)),
                cam_wrist
            ], axis=0)

        cam_wrist_future = self._gather_frames(wrist_frames, future_indices)
        if len(cam_wrist_future) == 0:
            cam_wrist_future = np.repeat(cam_wrist[-1:], len(future_indices), axis=0)

        # ── Load segmentation masks ──
        # input_mask controls the first-frame mask; predict_mask controls future masks.
        first_moving_mask = None
        seg_mask_future = None
        wrist_seg_mask_future = None

        if self.input_mask or self.predict_mask:
            _fh, _fw = cam_high.shape[1:3]
            need_future_masks = self.predict_mask

            # Build frame indices to read from front mask mp4
            front_mask_indices = []
            if self.input_mask:
                front_mask_indices.append(first_moving_idx)
            if need_future_masks:
                front_mask_indices.extend(future_indices)

            front_mask_frames = {}
            if front_mask_indices:
                front_mask_frames = self.parse_mask_video_all(
                    episode_dir / "cam_high_mask.mp4", front_mask_indices, num_steps
                )
                if not front_mask_frames:
                    return False, None

            # Current mask (first_moving_mask)
            if self.input_mask:
                if first_moving_idx not in front_mask_frames:
                    return False, None
                first_moving_mask = front_mask_frames[first_moving_idx]
                if first_moving_mask.shape[:2] != (_fh, _fw):
                    first_moving_mask = cv2.resize(first_moving_mask, (_fw, _fh))
                if np.all(first_moving_mask < 64):
                    return False, None
            else:
                first_moving_mask = np.zeros((_fh, _fw, 3), dtype=np.uint8)

            # Future masks
            if need_future_masks:
                seg_mask_future = self._gather_frames(front_mask_frames, future_indices)
                if len(seg_mask_future) == 0:
                    seg_mask_future = np.repeat(first_moving_mask[None], len(future_indices), axis=0)
                if seg_mask_future.shape[1:3] != cam_high_future.shape[1:3]:
                    h, w = cam_high_future.shape[1:3]
                    seg_mask_future = np.stack([cv2.resize(m, (w, h)) for m in seg_mask_future])

                wrist_mask_indices = list(future_indices)
                wrist_mask_frames = self.parse_mask_video_all(
                    episode_dir / "cam_left_wrist_mask.mp4", wrist_mask_indices, num_steps
                )
                if not wrist_mask_frames:
                    return False, None
                wrist_seg_mask_future = self._gather_frames(wrist_mask_frames, future_indices)
                if len(wrist_seg_mask_future) == 0:
                    wrist_seg_mask_future = np.repeat(
                        np.zeros((_fh, _fw, 3), dtype=np.uint8)[None], len(future_indices), axis=0
                    )
                if wrist_seg_mask_future.shape[1:3] != cam_wrist_future.shape[1:3]:
                    h, w = cam_wrist_future.shape[1:3]
                    wrist_seg_mask_future = np.stack([cv2.resize(m, (w, h)) for m in wrist_seg_mask_future])
            else:
                seg_mask_future = np.zeros(
                    (len(future_indices), *cam_high_future.shape[1:3], 3), dtype=np.uint8
                )
                wrist_seg_mask_future = np.zeros(
                    (len(future_indices), *cam_wrist_future.shape[1:3], 3), dtype=np.uint8
                )

        # ── Data augmentation (per-view, before T-shape concat) ──
        if self.image_aug:
            front_h, front_w = cam_high.shape[1:3]
            wrist_h, wrist_w = cam_wrist.shape[1:3]

            if first_moving_mask is None:
                first_moving_mask = np.zeros((front_h, front_w, 3), dtype=np.uint8)
            if seg_mask_future is None:
                seg_mask_future = np.zeros((len(future_indices), front_h, front_w, 3), dtype=np.uint8)
            if wrist_seg_mask_future is None:
                wrist_seg_mask_future = np.zeros((len(future_indices), wrist_h, wrist_w, 3), dtype=np.uint8)

            front_masks = np.repeat(first_moving_mask[None], cam_high.shape[0], axis=0)
            cam_high, front_masks, cam_high_future, seg_mask_future = self._apply_augmentation(
                cam_high, front_masks, cam_high_future, seg_mask_future
            )
            first_moving_mask = front_masks[-1]

            wrist_masks = np.zeros((cam_wrist.shape[0], wrist_h, wrist_w, 3), dtype=np.uint8)
            cam_wrist, wrist_masks, cam_wrist_future, wrist_seg_mask_future = self._apply_augmentation(
                cam_wrist, wrist_masks, cam_wrist_future, wrist_seg_mask_future
            )

        # ── Resize wrist to match front spatial dimensions before T-shape concat ──
        front_h, front_w = cam_high.shape[1:3]
        wrist_h, wrist_w = cam_wrist.shape[1:3]
        if (wrist_h, wrist_w) != (front_h, front_w):
            cam_wrist = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_wrist])
            cam_wrist_future = np.stack([cv2.resize(img, (front_w, front_h)) for img in cam_wrist_future])
            if wrist_seg_mask_future is not None:
                wrist_seg_mask_future = np.stack([cv2.resize(m, (front_w, front_h)) for m in wrist_seg_mask_future])

        # ── T-shape concatenation ──
        # Images: front + wrist → T-shape for both history and future
        t_cam_high = concat_t_shape_images_batch(cam_high, cam_wrist)
        t_cam_high_future = concat_t_shape_images_batch(cam_high_future, cam_wrist_future)

        # Masks (RGB): when input_mask or predict_mask
        if self.input_mask or self.predict_mask:
            black_wrist = np.zeros((front_h, front_w, 3), dtype=np.uint8)
            t_current_seg_mask = concat_t_shape_image(
                first_moving_mask, black_wrist
            )  # (H_T, W, 3) uint8
            t_seg_mask_future = concat_t_shape_images_batch(
                seg_mask_future, wrist_seg_mask_future
            )
        else:
            t_current_seg_mask = None
            t_seg_mask_future = None

        meta = {
            "dataset_name": dataset_name,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction": instruction,
            "instruction_idx": instruction_idx,
            "task": instruction,
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
            "current_image": t_cam_high[-1],
            "future_images": t_cam_high_future,
            "t5_embed": t5_embed,
        }
        if self.input_mask or self.predict_mask:
            result["current_seg_mask"] = t_current_seg_mask  # (H_T, W, 3) uint8 RGB
        if self.predict_mask:
            result["future_seg_masks"] = t_seg_mask_future  # (N, H_T, W, 3) uint8 RGB
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
