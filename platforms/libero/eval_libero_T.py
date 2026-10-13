#!/usr/bin/env python3
"""
MaskWAM Sim T-shape Action Policy Evaluation Script on LIBERO Benchmark

This script evaluates trained mask_wan_mot_sim policy models (T-shape variant) on
LIBERO benchmark tasks. It:
  - Uses ``mask_wan_mot_sim.MaskDitVideoPolicy`` which expects ``current_seg_mask``
    as an **RGB image** ``[B, 3, H, W] float∈[0,1]`` (red=object, blue=robot,
    black=background), matching the training pipeline
    (``data/hdf5_dataset_libero_T.py`` + ``convert_to_vam_step3.py``).
  - Concatenates front (agentview) + wrist (eye_in_hand) cameras into a T-shape
    image before feeding into the model as `current_image`.
  - Uses non-square video dimensions (video_height x video_width) from model config.
  - Wrist region in ``current_seg_mask`` is always BLACK (zeros), matching training.
  - The front mask is captured ONCE per episode after objects have stabilised
    (right before the policy starts acting), mirroring the ``first_moving_idx``
    convention used during training.

Usage:
    scripts/eval_libero.sh --task_suite libero_10 --num_trials 50
    scripts/eval_libero.sh --task_suite libero_10 --t5_embeds path/to/t5.pt
    # disable mask conditioning (zero RGB mask):
    scripts/eval_libero.sh --task_suite libero_10 --no_mask
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import sys
import gc
import faulthandler

# Print Python stack on any fatal signal (SIGSEGV/SIGABRT/SIGFPE/SIGBUS/SIGILL).
# Helps locate which Python call triggered a C-level crash (MuJoCo / CUDA / EGL).
faulthandler.enable(all_threads=True)

libero_root = os.environ.get("LIBERO_ROOT")
if libero_root:
    sys.path.insert(0, libero_root)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
import logging
import pickle
import cv2
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from collections import deque
from typing import Dict, Optional, Tuple
import imageio

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs import SegmentationRenderEnv

from checkpoint_loader import load_policy
from paths import CHECKPOINT_ROOT, DATA_ROOT, WAN_ROOT

# Robot-related keywords for matching instance names in LIBERO/robosuite.
# Uses substring matching to cover variants like Panda0, OnTheGroundPanda0,
# PandaGripper0, RethinkMount0, NullMount0, etc. (matches regenerate_robot_mask.py)
ROBOT_INSTANCE_KEYWORDS = ["Panda", "Mount"]

# ─── Helpers ───────────────────────────────────────────────────────────────────

def resize_with_pad(img, width, height, pad_value=0, mode="bilinear"):
    """Resize image tensor (B, C, H, W) with padding to maintain aspect ratio."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, got {img.shape}")
    cur_h, cur_w = img.shape[2:]
    ratio = max(cur_w / width, cur_h / height)
    rh, rw = max(1, int(round(cur_h / ratio))), max(1, int(round(cur_w / ratio)))
    kw = {"size": (rh, rw), "mode": mode}
    if mode != "nearest":
        kw["align_corners"] = False
    resized = F.interpolate(img, **kw)
    ph, pw = max(0, height - rh), max(0, width - rw)
    return F.pad(resized, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), value=pad_value)

def pad_vector(vector, new_dim):
    """Pad last dimension of vector to new_dim with zeros."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    shape[-1] = new_dim
    new_vector = np.zeros(shape, dtype=vector.dtype if hasattr(vector, 'dtype') else np.float32)
    new_vector[..., : vector.shape[-1]] = vector
    return new_vector

def convert_image_hw(img_np, target_h, target_w):
    """Convert RGB numpy image (H, W, 3) uint8 to tensor (3, target_h, target_w) in [0, 1]."""
    tensor = torch.from_numpy(img_np.copy()).permute(2, 0, 1).float() / 255.0
    tensor = resize_with_pad(tensor.unsqueeze(0), target_w, target_h, pad_value=0)[0]
    return tensor

def convert_rgb_mask_hw(mask_rgb_np, target_h, target_w):
    """Convert RGB mask image (H, W, 3) uint8 to tensor (3, target_h, target_w) float [0, 1].

    Uses ``mode='nearest'`` to preserve the categorical RGB palette
    (red=object, blue=robot, black=background), matching
    ``data/hdf5_dataset_libero_T.py::process_video_masks_rgb``.
    """
    if mask_rgb_np.ndim != 3 or mask_rgb_np.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB mask with shape (H, W, 3), got {mask_rgb_np.shape}"
        )
    tensor = torch.from_numpy(mask_rgb_np.copy()).permute(2, 0, 1).float() / 255.0
    tensor = resize_with_pad(
        tensor.unsqueeze(0), target_w, target_h, pad_value=0, mode="nearest"
    )[0]
    return tensor

# ─── Robot+object mask extraction (matches regenerate_robot_mask.py) ──────────

def build_robot_seg_ids(env):
    """Find all robot-related segmentation IDs after ``env.reset()``.

    Mirrors ``regenerate_robot_mask.build_robot_seg_ids``: instance index ``i``
    in ``env.env.model.instances_to_ids`` corresponds to pixel value ``i + 1``
    in the segmentation image.
    """
    robot_seg_values = set()
    for i, instance_name in enumerate(list(env.env.model.instances_to_ids.keys())):
        if any(kw in instance_name for kw in ROBOT_INSTANCE_KEYWORDS):
            robot_seg_values.add(i + 1)
    return robot_seg_values

def get_obj_robot_mask(env, segmentation_image, robot_seg_values):
    """Convert instance segmentation to a tri-value float mask {0.0, 0.5, 1.0}.

    1.0 = object of interest, 0.5 = robot, 0.0 = background. Matches
    ``regenerate_robot_mask.get_obj_robot_mask``.
    """
    mask = np.zeros(segmentation_image.shape, dtype=np.float32)
    for rv in robot_seg_values:
        mask[segmentation_image == rv] = 0.5
    for obj in env.obj_of_interest:
        if obj in env.instance_to_id:
            mask[segmentation_image == env.instance_to_id[obj]] = 1.0
    return mask

def mask_to_rgb_frame(mask: np.ndarray) -> np.ndarray:
    """Convert (H, W) tri-value float mask to (H, W, 3) uint8 RGB.

    Matches ``convert_to_vam.mask_to_rgb_frames``: object→red (255,0,0),
    robot→blue (0,0,255), background→black (0,0,0).
    """
    H, W = mask.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[mask >= 0.75, 0] = 255
    rgb[(mask >= 0.25) & (mask < 0.75), 2] = 255
    return rgb

# ─── T-shape concat (matches data/hdf5_dataset_libero_T.py) ───────────────────

def concat_t_shape_image(front_img: np.ndarray, wrist_img: np.ndarray) -> np.ndarray:
    """Concatenate front and wrist images into T-shape layout.

    Args:
        front_img: (H, W, 3) front camera image.
        wrist_img: (H, W, 3) left wrist camera image (will be duplicated as right wrist).

    Returns:
        T-shape concatenated image of shape (H + H//2, W, 3).
    """
    H, W = front_img.shape[:2]
    half_h, half_w = H // 2, W // 2
    left_resized = cv2.resize(wrist_img, (half_w, half_h))
    right_resized = left_resized.copy()
    bottom_row = np.hstack([left_resized, right_resized])
    return np.vstack([front_img, bottom_row])

# ─── Env / Seed / Logging ─────────────────────────────────────────────────────

def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def set_seed_everywhere(seed: int):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def create_libero_env(task, resolution=256, use_mask=False):
    """Create and initialize LIBERO environment.

    Args:
        task: LIBERO task object.
        resolution: Camera resolution.
        use_mask: If True, use SegmentationRenderEnv to get instance segmentation
                  masks from the simulator.
    """
    task_description = task.language
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )
    if use_mask:
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "camera_segmentations": "instance",
            # Pin MuJoCo's offscreen renderer to CPU/EGL device -1 to avoid
            # EGL/CUDA context conflicts that show up as "Aborted (core dumped)"
            # after the first episode. Mirrors the OffScreenRenderEnv branch.
            "renderer": "mujoco",
            "renderer_config": {"device_id": -1},
        }
        env = SegmentationRenderEnv(**env_args)
    else:
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": resolution,
            "camera_widths": resolution,
            "renderer": "mujoco",
            "renderer_config": {"device_id": -1}
        }
        env = OffScreenRenderEnv(**env_args)
    env.seed(42)
    return env, task_description

class VideoRecorder:
    """Video recorder for saving evaluation episodes."""

    def __init__(self, save_dir="eval_videos", fps=20):
        self.save_dir = Path(save_dir)
        self.fps = fps
        self.frames = []
        self.recording = False
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def start_recording(self, task_name, episode_idx):
        self.frames = []
        self.recording = True
        self.task_name = task_name.replace(" ", "_").replace("/", "_")
        self.episode_idx = episode_idx

    def add_frame(self, obs):
        if self.recording and 'agentview_image' in obs:
            frame = obs['agentview_image'][::-1, :].copy()
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            self.frames.append(frame)

    def stop_recording(self, success=False):
        if not self.recording or len(self.frames) == 0:
            return None
        self.recording = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "fail"
        filename = f"{self.task_name}_ep{self.episode_idx}_{status}_{timestamp}.mp4"
        filepath = self.save_dir / filename
        try:
            imageio.mimsave(str(filepath), self.frames, fps=self.fps)
            return str(filepath)
        except Exception as e:
            logging.error(f"Failed to save video: {e}")
            return None

# ─── Evaluator ─────────────────────────────────────────────────────────────────

class LiberoEvaluator:
    """Main evaluator for LIBERO benchmark with MaskWAM sim T-shape models."""

    def __init__(self, args):
        self.args = args
        self.logger = setup_logging()
        self.weight_dtype = torch.bfloat16

        set_seed_everywhere(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        # Load model
        self.setup_model()

        # Load normalization stats
        self.setup_normalization()

        # Setup T5 encoder / embeddings
        self.setup_t5()

        # Setup mask detection (if use_mask)
        self.setup_mask()

        # Setup task suite
        self.setup_task_suite()

        # Video recorder
        self.video_recorder = None
        if args.record_video:
            self.video_recorder = VideoRecorder(save_dir=args.video_dir)

        # Results storage
        self.results = {
            'task_results': {},
            'overall_success_rate': 0.0,
            'total_episodes': 0,
            'total_successes': 0
        }

    # ── Model loading ─────────────────────────────────────────────────────

    def setup_model(self):
        """Load the MaskWAM sim T-shape model."""
        self.logger.info(f"Loading MaskDitVideoPolicy (T-shape) from: {self.args.ckpt_path}")
        self.policy, _ = load_policy(
            self.args.ckpt_path,
            device=str(self.device),
            dtype="bfloat16",
            wan_checkpoint_path=self.args.wan_path,
        )

        # Read video dimensions from model config (T-shape: 384x320)
        self.video_height = self.policy.config.video_height
        self.video_width = self.policy.config.video_width
        self.max_state_dim = self.policy.config.max_state_dim
        self.max_action_dim = self.policy.config.max_action_dim
        self.action_dim = self.policy.config.action_feature.shape[0]

        self.logger.info(
            f"  T-shape input (H, W) = ({self.video_height}, {self.video_width}), "
            f"use_mask={self.policy.model.use_mask}, "
            f"predict_mask={self.policy.model.predict_mask}, "
            f"n_action_steps={self.policy.config.n_action_steps}, "
            f"action_dim={self.action_dim}, "
            f"max_state_dim={self.max_state_dim}"
        )

    # ── Mask setup ─────────────────────────────────────────────────────────

    def setup_mask(self):
        """Setup mask configuration for the LIBERO segmentation environment.

        When use_mask=True, the environment is created with SegmentationRenderEnv
        and we extract a tri-value object+robot mask (1.0=obj, 0.5=robot, 0.0=bg)
        from the agentview segmentation image, then encode it as RGB
        (red=object, blue=robot, black=bg) — exactly matching the training
        pipeline (regenerate_robot_mask.py + convert_to_vam.py +
        hdf5_dataset_libero_T.py). The wrist region in the T-shape
        ``current_seg_mask`` is always BLACK, also matching training.
        """
        self.use_mask = (
            self.args.use_mask
            and self.policy.model.use_mask
            and self.policy.model.input_mask
        )
        # Per-episode cache of robot instance segmentation IDs (rebuilt after env.reset()).
        self._robot_seg_values = set()

        if self.use_mask:
            self.logger.info(
                "  Mask enabled: extracting tri-value object+robot mask from "
                "LIBERO SegmentationRenderEnv (encoded as RGB: red=obj, "
                "blue=robot, black=bg). Wrist region is always BLACK."
            )
        else:
            self.logger.info(
                "  Input mask disabled by CLI or checkpoint config; "
                "feeding zero RGB mask if the model still expects one."
            )

    def _get_front_mask_from_obs(self, env, obs) -> Optional[np.ndarray]:
        """Extract front (agentview) tri-value object+robot mask from observation.

        Returns:
            (H, W) float32 mask in {0.0, 0.5, 1.0} (1=obj, 0.5=robot, 0=bg),
            flipped to match training convention. None if segmentation is missing.
        """
        seg_key = "agentview_segmentation_instance"
        if seg_key not in obs:
            return None
        agentview_seg = obs[seg_key].squeeze(-1)[::-1, :]
        return get_obj_robot_mask(env, agentview_seg, self._robot_seg_values)

    # ── Normalization ─────────────────────────────────────────────────────

    def setup_normalization(self):
        """Load normalization statistics."""
        self.has_norm = False
        self.qpos_mean = self.qpos_std = None
        self.act_mean = self.act_std = None

        if self.args.mean_std_path and os.path.exists(self.args.mean_std_path):
            with open(self.args.mean_std_path, "rb") as f:
                stats = pickle.load(f)
            self.act_mean = np.array(stats["action_mean"], dtype=np.float32)
            self.act_std = np.array(stats["action_std"], dtype=np.float32)
            self.qpos_mean = np.array(stats["qpos_mean"], dtype=np.float32)
            self.qpos_std = np.array(stats["qpos_std"], dtype=np.float32)
            self.has_norm = True
            self.logger.info(f"  Loaded normalization stats from {self.args.mean_std_path}")
        else:
            self.logger.warning("  No normalization stats loaded. Using raw values.")

    # ── T5 encoder ────────────────────────────────────────────────────────

    def setup_t5(self):
        """Setup T5 embeddings.

        Three accepted formats for ``--t5_embeds``:
            1. ``dict`` payload from ``precompute_t5_libero.py``::
                   {"embeddings": {instruction_str: Tensor[L, D]}, "task_index": [...]}
               Looked up by instruction string at runtime — preferred.
            2. Plain ``{instruction_str: Tensor}`` dict.
            3. Single ``Tensor`` (or ``list[Tensor]`` of length 1) used for
               every instruction (legacy single-task behaviour).
        Falls back to loading the WAN T5 encoder on-the-fly when no file is
        provided (slow, ~11GB).
        """
        self.t5_encoder = None
        self.cached_t5_embeds: Dict[str, torch.Tensor] = {}
        self._default_t5_embed = None

        if self.args.t5_embeds and os.path.exists(self.args.t5_embeds):
            loaded = torch.load(self.args.t5_embeds, map_location="cpu")
            if isinstance(loaded, dict) and "embeddings" in loaded and isinstance(
                loaded["embeddings"], dict
            ):
                # precompute_t5_libero.py format
                self.cached_t5_embeds = {
                    k: v if torch.is_tensor(v) else torch.as_tensor(v)
                    for k, v in loaded["embeddings"].items()
                }
                self.logger.info(
                    f"  Loaded {len(self.cached_t5_embeds)} pre-encoded T5 "
                    f"embeddings from {self.args.t5_embeds}"
                )
            elif isinstance(loaded, dict):
                # Plain {instruction: tensor} dict
                self.cached_t5_embeds = {
                    k: v if torch.is_tensor(v) else torch.as_tensor(v)
                    for k, v in loaded.items()
                }
                self.logger.info(
                    f"  Loaded {len(self.cached_t5_embeds)} pre-encoded T5 "
                    f"embeddings (dict) from {self.args.t5_embeds}"
                )
            elif isinstance(loaded, torch.Tensor):
                self._default_t5_embed = loaded
                self.logger.info(
                    f"  Loaded single T5 embedding tensor from {self.args.t5_embeds} "
                    "(will be reused for every instruction)"
                )
            elif isinstance(loaded, list):
                self._default_t5_embed = loaded[0] if len(loaded) == 1 else loaded
                self.logger.info(
                    f"  Loaded T5 embedding list ({len(loaded)}) from {self.args.t5_embeds}"
                )
            else:
                raise TypeError(
                    f"Unsupported --t5_embeds payload type: {type(loaded)}"
                )
        else:
            self.logger.warning(
                "  No --t5_embeds provided; falling back to on-the-fly T5 "
                "encoder (slow, requires ~11GB GPU memory). Consider running "
                "platforms/libero/precompute_t5_libero.py first."
            )
            self._init_t5_encoder()

    def _init_t5_encoder(self):
        """Initialize T5 encoder for on-the-fly text encoding."""
        import traceback

        wan_path = self.args.wan_path
        # Resolve relative paths against the wam project root to avoid surprises
        # when the script is launched from a different cwd.
        if not os.path.isabs(wan_path):
            wan_path = os.path.abspath(os.path.join(parent_dir, wan_path))

        t5_ckpt = os.path.join(wan_path, "models_t5_umt5-xxl-enc-bf16.pth")
        t5_tokenizer = os.path.join(wan_path, "google/umt5-xxl")

        # Surface missing files immediately with a clear message.
        if not os.path.exists(t5_ckpt):
            raise FileNotFoundError(
                f"T5 checkpoint not found: {t5_ckpt} (wan_path={wan_path}). "
                f"Pass --wan_path or provide --t5_embeds."
            )
        if not os.path.exists(t5_tokenizer):
            raise FileNotFoundError(
                f"T5 tokenizer dir not found: {t5_tokenizer} (wan_path={wan_path}). "
                f"Pass --wan_path or provide --t5_embeds."
            )

        try:
            from wan22_core.bak.wan.modules.t5 import T5EncoderModel
            self.t5_encoder = T5EncoderModel(
                text_len=512,
                dtype=torch.bfloat16,
                device="cuda",
                checkpoint_path=t5_ckpt,
                tokenizer_path=t5_tokenizer,
            )
            self.logger.info(f"  Initialized T5 encoder from {wan_path}")
        except Exception as e:
            self.logger.error(f"  Failed to initialize T5 encoder: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def _get_t5_embed(self, instruction: str) -> torch.Tensor:
        """Get T5 embedding for an instruction.

        Lookup priority:
            1. ``cached_t5_embeds`` (from precomputed file or previous on-the-fly).
            2. ``_default_t5_embed`` (single-tensor / list payload).
            3. on-the-fly encoder (must be initialized).
        """
        if instruction in self.cached_t5_embeds:
            return self.cached_t5_embeds[instruction].to(device="cuda", dtype=self.weight_dtype)

        if self._default_t5_embed is not None:
            return self._default_t5_embed.to(device="cuda", dtype=self.weight_dtype)

        if self.t5_encoder is None:
            available = list(self.cached_t5_embeds.keys())
            raise RuntimeError(
                f"No T5 embedding for instruction: {instruction!r}. "
                f"Either pass --t5_embeds with a payload that includes this "
                f"instruction (precompute via platforms/libero/precompute_t5_libero.py), "
                f"or omit --t5_embeds to fall back to the on-the-fly T5 encoder. "
                f"Currently cached: {len(available)} instructions."
            )

        with torch.no_grad():
            t5_out = self.t5_encoder([instruction], "cuda")
        if isinstance(t5_out, list):
            embed = t5_out[0]
        elif isinstance(t5_out, torch.Tensor):
            embed = t5_out.squeeze(0) if t5_out.ndim == 3 else t5_out
        else:
            raise ValueError(f"Unexpected T5 output type: {type(t5_out)}")

        self.cached_t5_embeds[instruction] = embed.detach().cpu()
        return embed.to(device="cuda", dtype=self.weight_dtype)

    # ── T-shape image/mask building ───────────────────────────────────────

    def build_t_shape_inputs(self, front_img_np, wrist_img_np, front_seg_mask):
        """Build T-shape image and RGB mask from front + wrist inputs.

        1. Resize wrist to match front size.
        2. T-shape concat for the image: front on top, wrist halved on bottom.
        3. T-shape concat for the mask as a 3-channel RGB image, with the
           wrist region always BLACK (matching ``hdf5_dataset_libero_T``
           ``current_seg_mask`` construction).

        Args:
            front_img_np: (H, W, 3) front camera image (already flipped).
            wrist_img_np: (H, W, 3) wrist camera image (already flipped), or None.
            front_seg_mask: (H, W) front tri-value mask in {0, 0.5, 1.0}, or None.

        Returns:
            t_img: (T_H, T_W, 3) T-shape concatenated RGB image.
            t_mask_rgb: (T_H, T_W, 3) uint8 T-shape RGB mask, or None.
        """
        # Resize wrist to match front spatial dimensions
        front_h, front_w = front_img_np.shape[:2]
        if wrist_img_np is not None:
            wh, ww = wrist_img_np.shape[:2]
            if (wh, ww) != (front_h, front_w):
                wrist_img_np = cv2.resize(wrist_img_np, (front_w, front_h))
        else:
            wrist_img_np = np.zeros_like(front_img_np)

        # T-shape concat for the image
        t_img = concat_t_shape_image(front_img_np, wrist_img_np)

        # T-shape concat for the RGB mask: encode tri-value front mask to RGB,
        # then use the same image-style concat with a black wrist patch.
        t_mask_rgb = None
        if front_seg_mask is not None:
            front_mask_rgb = mask_to_rgb_frame(front_seg_mask.astype(np.float32))
            black_wrist = np.zeros((front_h, front_w, 3), dtype=np.uint8)
            t_mask_rgb = concat_t_shape_image(front_mask_rgb, black_wrist)

        return t_img, t_mask_rgb

    # ── Observation → batch conversion ────────────────────────────────────

    def convert_obs_to_batch(self, obs, task_description: str,
                             seg_mask: Optional[np.ndarray] = None) -> dict:
        """Convert LIBERO observation to MaskWAM sim T-shape batch format.

        MaskWAM sim T-shape expects:
            - 'observation.state': (1, max_state_dim) float tensor
            - 'current_image': (1, 3, video_height, video_width) float tensor in [0, 1]
            - 't5_embed': list of T5 embedding tensors
            - 'current_seg_mask': (optional) (1, video_height, video_width) float tensor

        Args:
            obs: LIBERO observation dict
            task_description: task instruction string
            seg_mask: (H, W) front binary mask from segmentation (or None for zero mask)
        """
        # ── Front camera (agentview) ──
        front_np = obs['agentview_image'][::-1, :].copy()

        # ── Wrist camera (eye_in_hand) ──
        if 'robot0_eye_in_hand_image' in obs:
            wrist_np = obs['robot0_eye_in_hand_image'][::-1, :].copy()
        else:
            self.logger.warning("robot0_eye_in_hand_image not found in obs; using zeros.")
            wrist_np = None

        # ── Build T-shape image and RGB mask ──
        t_img, t_mask_rgb = self.build_t_shape_inputs(front_np, wrist_np, seg_mask)
        current_image = convert_image_hw(t_img, self.video_height, self.video_width)

        # ── State: gripper_qpos(2) + eef_pos(3) + eef_quat(4) = 9 dim ──
        robot_state = np.concatenate([
            obs["robot0_gripper_qpos"],
            obs["robot0_eef_pos"],
            obs["robot0_eef_quat"]
        ]).astype(np.float32)

        if self.has_norm:
            robot_state = (robot_state - self.qpos_mean) / self.qpos_std

        padded_state = pad_vector(robot_state[np.newaxis, :], self.max_state_dim).astype(np.float32)

        # ── T5 embedding ──
        t5_embed = self._get_t5_embed(task_description)

        # ── Build batch ──
        batch = {
            "observation.state": torch.from_numpy(padded_state).cuda().to(self.weight_dtype),
            "current_image": current_image.unsqueeze(0).cuda().to(self.weight_dtype),
            "t5_embed": [t5_embed],
        }

        # ── Segmentation mask (RGB, [1, 3, H_T, W_T]) ──
        if self.policy.model.use_mask:
            if t_mask_rgb is not None:
                current_seg_mask = convert_rgb_mask_hw(
                    t_mask_rgb, self.video_height, self.video_width
                )
                batch["current_seg_mask"] = (
                    current_seg_mask.unsqueeze(0).cuda().to(self.weight_dtype)
                )
            else:
                zero_mask = torch.zeros(
                    1, 3, self.video_height, self.video_width,
                    device="cuda", dtype=self.weight_dtype,
                )
                batch["current_seg_mask"] = zero_mask

        return batch

    # ── Task suite ────────────────────────────────────────────────────────

    def setup_task_suite(self):
        """Setup LIBERO task suite."""
        benchmark_dict = benchmark.get_benchmark_dict()
        self.task_suite = benchmark_dict[self.args.task_suite]()
        self.num_tasks = self.task_suite.n_tasks

        self.logger.info(f"Task suite: {self.args.task_suite}")
        self.logger.info(f"Number of tasks: {self.num_tasks}")

        max_steps_map = {
            "libero_spatial": 400,
            "libero_object": 400,
            "libero_goal": 400,
            "libero_10": 700,
            "libero_90": 400,
        }
        self.max_steps = max_steps_map.get(self.args.task_suite, 300)
        self.logger.info(f"Max steps per episode: {self.max_steps}")

    # ── Action processing ─────────────────────────────────────────────────

    def process_action(self, action):
        """Process raw model action → LIBERO 7-dim action.

        Model outputs (max_action_dim,) normalized action.
        LIBERO needs 7-dim: [delta_pos(3), delta_euler(3), gripper(1)].
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy().squeeze()
        elif isinstance(action, np.ndarray):
            action = action.squeeze()

        # Take first 7 dims (LIBERO action space)
        action = action[:7].copy()

        # Denormalize
        if self.has_norm:
            action = action * self.act_std + self.act_mean

        # Binarize gripper
        action[-1] = np.sign(action[-1])

        return action.tolist()

    # ── Episode execution ─────────────────────────────────────────────────

    def run_episode(self, env, task_description, episode_idx, obs):
        """Run a single episode."""
        num_steps_wait = 30
        action_queue = deque()
        t = 0
        episode_mask = None  # Cached front mask for this episode (from first stable frame)

        while t < self.max_steps + num_steps_wait:
            # Wait for objects to settle
            if t < num_steps_wait:
                obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                if self.video_recorder:
                    self.video_recorder.add_frame(obs)
                t += 1

                # Extract front mask on the LAST stable frame (right before action starts)
                if t == num_steps_wait and self.use_mask:
                    episode_mask = self._get_front_mask_from_obs(env, obs)
                    if episode_mask is not None:
                        n_obj = int((episode_mask >= 0.75).sum())
                        n_robot = int(((episode_mask >= 0.25) & (episode_mask < 0.75)).sum())
                        self.logger.info(
                            f"  Episode {episode_idx}: front mask extracted from stable frame "
                            f"(obj_px={n_obj}, robot_px={n_robot})"
                        )
                    else:
                        self.logger.warning(
                            f"  Episode {episode_idx}: front mask extraction FAILED, using zero RGB mask"
                        )
                continue

            # Use cached actions if available
            if len(action_queue) > 0:
                libero_action = self.process_action(action_queue.popleft())
                obs, reward, done, info = env.step(libero_action)
                if self.video_recorder:
                    self.video_recorder.add_frame(obs)
                if done or info.get('success', False):
                    return True
                t += 1
                continue

            # Generate new action chunk (pass episode_mask to batch)
            batch = self.convert_obs_to_batch(obs, task_description, seg_mask=episode_mask)
            self.policy.reset()

            action0 = self.policy.select_action(batch, save_video_latent=False)
            # Collect remaining actions from the queue
            remaining = []
            while self.policy._action_queue:
                remaining.append(self.policy._action_queue.popleft())

            # Execute first action
            libero_action = self.process_action(action0)
            obs, reward, done, info = env.step(libero_action)
            if self.video_recorder:
                self.video_recorder.add_frame(obs)
            if done or info.get('success', False):
                return True
            t += 1

            # Cache remaining actions (use action_chunk_size)
            chunk_size = self.args.action_chunk_size
            for i, act in enumerate(remaining):
                if i >= chunk_size - 1:  # -1 because action0 already executed
                    break
                action_queue.append(act)

        return False

    # ── Task evaluation ───────────────────────────────────────────────────

    def evaluate_task(self, task_id):
        """Evaluate a single task."""
        task = self.task_suite.get_task(task_id)
        initial_states = self.task_suite.get_task_init_states(task_id)

        # Resolve task description once (cheap, doesn't need a live env).
        task_description = task.language
        self.logger.info(f"\nTask {task_id}: {task_description}")

        # 是否对本任务录像：每 video_every 个任务录一个（按全局 task_id 取模）。
        should_record = (
            self.video_recorder is not None
            and self.args.video_every > 0
            and task_id % self.args.video_every == 0
        )

        task_episodes = 0
        task_successes = 0

        pbar = tqdm(range(self.args.num_trials), desc=f'Task {task_id}')

        for episode_idx in pbar:
            # Drop any leftover GPU tensors from the previous episode
            # (stale entries in policy._action_queue when an episode terminates
            # early, lingering autograd graph fragments, etc.).
            self.policy.reset()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                self.logger.info(
                    f"  [pre ep{episode_idx}] GPU mem alloc="
                    f"{torch.cuda.memory_allocated()/1e9:.2f}GB "
                    f"reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
                    f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB"
                )
                torch.cuda.reset_peak_memory_stats()

            # Re-create env per episode so MuJoCo's EGL/GL render context is
            # fully torn down between episodes. This avoids the C-level segfault
            # in robosuite's binding_utils.read_pixels that surfaces when the
            # same env is reset multiple times with offscreen rendering.
            env, _ = create_libero_env(
                task, resolution=self.args.resolution, use_mask=self.use_mask
            )
            try:
                env.reset()
                # Rebuild robot segmentation IDs after each reset
                # (instances_to_ids is populated by env.reset() and may differ
                # across episodes).
                if self.use_mask:
                    self._robot_seg_values = build_robot_seg_ids(env)
                obs = env.set_init_state(initial_states[episode_idx])

                if should_record:
                    self.video_recorder.start_recording(
                        f"task{task_id}_{task_description}", episode_idx
                    )

                success = self.run_episode(env, task_description, episode_idx, obs)

                if should_record:
                    video_path = self.video_recorder.stop_recording(success)
                    if video_path:
                        self.logger.info(
                            f"Saved video ({'success' if success else 'fail'}): {video_path}"
                        )

                task_episodes += 1
                if success:
                    task_successes += 1

                current_success_rate = task_successes / task_episodes
                pbar.set_postfix({'success_rate': f'{current_success_rate:.2%}'})
            finally:
                try:
                    env.close()
                except Exception as e:
                    self.logger.warning(f"  env.close() failed: {e}")
                del env
                gc.collect()

        task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0.0
        try:
            bddl_name = os.path.basename(self.task_suite.get_task(task_id).bddl_file)
        except Exception:
            bddl_name = None
        self.results['task_results'][task_id] = {
            'task_description': task_description,
            'bddl_file': bddl_name,
            'episodes': task_episodes,
            'successes': task_successes,
            'success_rate': task_success_rate
        }

        self.logger.info(
            f"Task {task_id} completed: {task_successes}/{task_episodes} "
            f"({task_success_rate:.2%})"
        )

        return task_episodes, task_successes

    # ── Full evaluation ───────────────────────────────────────────────────

    def evaluate(self, task_start=0, task_end=None):
        """Run evaluation over a range of tasks (defaults to all tasks)."""
        if task_end is None:
            task_end = self.num_tasks
        task_start = max(0, task_start)
        task_end = min(self.num_tasks, task_end)
        self._eval_range = (task_start, task_end)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting T-shape Evaluation (tasks [{task_start}, {task_end}))")
        self.logger.info(f"{'='*60}\n")

        total_episodes = 0
        total_successes = 0

        for task_id in range(task_start, task_end):
            episodes, successes = self.evaluate_task(task_id)
            total_episodes += episodes
            total_successes += successes

        overall_success_rate = total_successes / total_episodes if total_episodes > 0 else 0.0

        self.results['total_episodes'] = total_episodes
        self.results['total_successes'] = total_successes
        self.results['overall_success_rate'] = overall_success_rate

        self.print_summary()
        self.save_results()
        return overall_success_rate

    def print_summary(self):
        """Print evaluation summary."""
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"T-shape Evaluation Summary")
        self.logger.info(f"{'='*60}\n")

        self.logger.info(f"Task Suite: {self.args.task_suite}")
        self.logger.info(f"Video Size: {self.video_height}x{self.video_width}")
        self.logger.info(f"Total Episodes: {self.results['total_episodes']}")
        self.logger.info(f"Total Successes: {self.results['total_successes']}")
        self.logger.info(f"Overall Success Rate: {self.results['overall_success_rate']:.2%}\n")

        self.logger.info("Per-Task Results:")
        self.logger.info("-" * 60)
        for task_id, result in self.results['task_results'].items():
            self.logger.info(
                f"Task {task_id}: {result['task_description']}\n"
                f"  Success: {result['successes']}/{result['episodes']} "
                f"({result['success_rate']:.2%})"
            )
        self.logger.info(f"{'='*60}\n")

    def save_results(self):
        """Save evaluation results to file."""
        import json

        ckpt_path_parts = Path(self.args.ckpt_path).parts
        ckpt_name = "unknown"
        for part in ckpt_path_parts:
            if part.isdigit():
                ckpt_name = part
                break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(self.args.results_dir) / "mask_wan_mot_sim" / ckpt_name
        results_dir.mkdir(parents=True, exist_ok=True)

        if self.args.task_id is not None:
            task_suffix = f"_task{self.args.task_id}"
        elif getattr(self, "_eval_range", None) is not None:
            task_suffix = f"_task{self._eval_range[0]}-{self._eval_range[1]}"
        else:
            task_suffix = ""
        results_file = results_dir / f"results_{self.args.task_suite}{task_suffix}_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        self.logger.info(f"Results saved to: {results_file}")

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MaskWAM sim T-shape on LIBERO benchmark")

    # Model arguments
    parser.add_argument("--ckpt_path", type=str,
                        default=str(CHECKPOINT_ROOT / "MaskWAM-libero-ckpt"),
                        help="Path to model checkpoint (the 'model' directory)")
    parser.add_argument("--wan_path", type=str,
                        default=str(WAN_ROOT),
                        help="Path to WAN pretrained models (for T5 encoder)")
    parser.add_argument("--t5_embeds", type=str, default=None,
                        help="Path to pre-encoded T5 embeddings (.pt file). "
                             "If not provided, T5 encoder will be initialized on-the-fly.")

    # Normalization
    parser.add_argument("--mean_std_path", type=str,
                        default=str(DATA_ROOT / "libero_vam" / "qpos_mean_std_online.pkl"),
                        help="Path to normalization statistics pickle file")

    # Task arguments
    parser.add_argument("--task_suite", type=str, default="libero_10",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10",
                                 "libero_90", "libero_100", "libero_mix"])
    parser.add_argument("--num_trials", type=int, default=50,
                        help="Number of trials per task")
    parser.add_argument("--task_id", type=int, default=None,
                        help="If set, only evaluate this specific task ID")
    parser.add_argument("--task_start", type=int, default=None,
                        help="Lower bound (inclusive) of task ID range to evaluate. "
                             "Use with --task_end to split a suite across GPUs/processes.")
    parser.add_argument("--task_end", type=int, default=None,
                        help="Upper bound (exclusive) of task ID range. Defaults to n_tasks.")

    # Action chunking
    parser.add_argument(
        "--action_chunk_size",
        type=int,
        default=16,
        help="Number of the model's 48 predicted actions to execute before re-planning",
    )

    # Mask arguments
    parser.add_argument("--use_mask", action="store_true", default=True,
                        help="Use LIBERO SegmentationRenderEnv to get first-frame front object mask "
                             "(default: True if model supports mask)")
    parser.add_argument("--no_mask", dest="use_mask", action="store_false",
                        help="Disable mask detection (use zero mask)")

    # Environment arguments
    parser.add_argument("--resolution", type=int, default=256,
                        help="Camera resolution")
    parser.add_argument("--seed", type=int, default=42)

    # Recording arguments
    parser.add_argument("--record_video", action="store_true", default=False)
    parser.add_argument("--video_dir", type=str, default="eval_videos")
    parser.add_argument("--video_every", type=int, default=1,
                        help="只对 task_id %% N == 0 的任务录视频（N=1 即每个任务都录）。")

    # Output arguments
    parser.add_argument("--results_dir", type=str, default="eval_results")

    args = parser.parse_args()

    evaluator = LiberoEvaluator(args)

    if args.task_id is not None:
        episodes, successes = evaluator.evaluate_task(args.task_id)
        evaluator.results['total_episodes'] = episodes
        evaluator.results['total_successes'] = successes
        evaluator.results['overall_success_rate'] = successes / episodes if episodes > 0 else 0.0
        evaluator.print_summary()
        evaluator.save_results()
    elif args.task_start is not None or args.task_end is not None:
        start = args.task_start if args.task_start is not None else 0
        end = args.task_end if args.task_end is not None else evaluator.num_tasks
        evaluator.evaluate(task_start=start, task_end=end)
    else:
        evaluator.evaluate()

if __name__ == "__main__":
    main()
