"""
Configuration for the simulation MaskWAM policy.

A dedicated, minimal config class that only contains fields
actually used by MaskWanBackbone / MaskDitVideoFlowMatching / MaskDitVideoPolicy.
"""

import types
import sys
from dataclasses import dataclass, field

import lerobot
from pathlib import Path

try:
    version = lerobot.__version__
except AttributeError:
    version = "unknown"

if version == "0.1.0":
    from lerobot.common.optim.optimizers import AdamWConfig
    from lerobot.common.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
elif version == "0.4.2":
    from lerobot.optim.optimizers import AdamWConfig
    from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


@PreTrainedConfig.register_subclass("mask_wan_mot_sim")
@dataclass
class MaskWanMotConfig(PreTrainedConfig):
    """Minimal config for the simulation MaskWAM policy."""

    # ── Input / output structure ──
    n_obs_steps: int = 1
    chunk_size: int = 48 #16
    n_action_steps: int = 48 #16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    action_feature: PolicyFeature = PolicyFeature(type=FeatureType.ACTION, shape=(14,))

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Projector
    proj_width: int = 1024

    # Decoding (action Euler steps)
    num_steps: int = 10

    # ── Video settings ──
    num_video_frames: int = 8
    video_height: int = 384
    video_width: int = 320
    batch_size: int = 4

    # ── WAN pretrained paths ──
    wan_checkpoint_path: str = "pretrained_models/Motus_Wan2_2_5B_pretrain"
    wan_config_path: str = "pretrained_models/Motus_Wan2_2_5B_pretrain"
    wan_vae_path: str = "pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
    wan_precision: str = "bfloat16"

    # ── Mask settings (Aether-style) ──
    use_mask: bool = True       # Legacy flag; kept for backward compatibility.
    # input_mask: controls the *source* of the first-frame mask condition.
    #   True  → condition_mask_latent = encode(current_seg_mask)
    #   False → condition_mask_latent = zeros (Aether-style "no mask signal")
    # Does NOT control DiT channel count; that is decided by (input_mask OR predict_mask).
    input_mask: bool = False
    # predict_mask: whether to predict mask (output head 48→96) and compute mask loss.
    # When True, mask stream is forced into DiT input (96ch) regardless of input_mask.
    predict_mask: bool = False
    mask_random: float = 1.0    # Probability of keeping current_seg_mask during training (modality masking)

    # ── Loss weights ──
    action_loss_weight: float = 1.0
    video_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0

    # ── Action flow-matching ──
    action_snr_shift: float = 1.0

    # ── Training presets ──
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Image preprocessing (used by dataset/collator)
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    device: str = "cpu"

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) must be <= chunk_size ({self.chunk_size})."
            )

    def validate_features(self) -> None:
        pass

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
