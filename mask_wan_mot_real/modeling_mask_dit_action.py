"""
mask_wan_mot_real policy: real/XTrainer mask-conditioned video generation.

Key flags:
  - `input_mask` (bool): first-frame mask condition SOURCE
      True  → condition_mask_latent = encode(current_seg_mask)
      False → condition_mask_latent = zeros_like(condition_frame_latent)
              (Aether-style "no mask signal" condition; skips a VAE encode)
  - `predict_mask` (bool): whether mask is a flow-matching target
      True  → output head is 96ch, mask loss enabled
      False → output head is 48ch, no mask loss

DiT input channel count = 48 + 48 * int(input_mask OR predict_mask).
Rationale: when predict_mask=True, x_t^mask MUST be visible in the DiT input;
otherwise mask velocity is unlearnable (target contains ε_mask but input does not).

Four valid combinations:
  (input_mask, predict_mask):
    (True,  True)  → mask IN (real) + mask OUT, 96/96  — original MaskWanMot
    (False, True)  → mask IN (zero) + mask OUT, 96/96  — Aether-style, new
    (True,  False) → mask IN (real) only,       96/48  — static mask condition
    (False, False) → no mask IO,                48/48  — degrades to wan_mot

Trainable components:
  - WAN transformer backbone inside video_module
  - DitActionExpert
  - action/state projection heads

Frozen components:
  - WAN VAE encoder/decoder used by encode_video / decode_video
"""

import logging
import math
import os
import sys

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from collections import deque
from PIL import Image
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from flash_attn import flash_attn_func as _flash_attn_func
import types
import numpy as np

def _fa2(q, k, v, softmax_scale=None):
    """Flash Attention 2 wrapper that ensures bf16 dtype.

    WAN's AdaLN operates in fp32, so QKV may arrive as fp32.
    flash_attn_func strictly requires fp16/bf16, so we cast here
    (matching WAN's own flash_attention wrapper which does the same).
    """
    dtype = torch.bfloat16
    return _flash_attn_func(
        q.to(dtype), k.to(dtype), v.to(dtype), softmax_scale=softmax_scale,
    )
import lerobot
from pathlib import Path

from .configuration_mask_wan_mot import MaskWanMotConfig
from .action_expert import DitActionExpert
from wan22_core.models.wan_model import WanVideoModel, _strip_known_prefixes_for_wan
from wan22_core.models.core import VideoModule

try:
    version = lerobot.__version__
except AttributeError:
    version = "unknown"

if version == "0.1.0":
    from lerobot.common.policies.pretrained import PreTrainedPolicy
    from lerobot.common.constants import ACTION
    from lerobot.common.constants import OBS_ROBOT as OBS_STATE
elif version == "0.4.2":
    policy_path = Path(lerobot.__file__).parent / "policies"
    m = types.ModuleType("lerobot.policies")
    m.__path__ = [str(policy_path)]
    sys.modules["lerobot.policies"] = m
    import lerobot.policies.pretrained as pretrained
    PreTrainedPolicy = pretrained.PreTrainedPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir    = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, root_dir)

from wan22_core.bak.wan.modules.model import rope_apply, Head
from wan22_core.bak.wan.utils.fm import FlowMatchScheduler
from wan22_core.bak.wan.utils.utils import masks_like
from wan22_core.bak.wan.utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)


# ── Shared utility functions ─────────────────────────────────────────────────

def pad_vector(vector, new_dim):
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def mask_to_red_rgb(mask: torch.Tensor) -> torch.Tensor:
    """Convert binary mask [B, 1, T, H, W] to red-RGB [B, 3, T, H, W].

    Object region (mask=1) → red (1, 0, 0); background (mask=0) → black (0, 0, 0).
    Output is in [-1, 1] range (normalized for VAE input).
    """
    red = mask
    green = torch.zeros_like(mask)
    blue = torch.zeros_like(mask)
    rgb = torch.cat([red, green, blue], dim=1)  # [B, 3, T, H, W] in [0, 1]
    rgb = rgb * 2.0 - 1.0
    return rgb


# ── Mask helpers (from MaskVDM) ──────────────────────────────────────────────

def prepare_current_mask(
    current_seg_mask: Optional[torch.Tensor], batch_size: int
) -> Optional[torch.Tensor]:
    """Normalize current mask to [B, 1, H, W]."""
    if current_seg_mask is None:
        return None
    if current_seg_mask.dim() == 3:
        current_seg_mask = current_seg_mask.unsqueeze(1)
    elif current_seg_mask.dim() == 4 and current_seg_mask.shape[1] == 1:
        pass
    else:
        raise ValueError(
            f"Expected current_seg_mask [B, H, W] or [B, 1, H, W], got {tuple(current_seg_mask.shape)}"
        )
    if current_seg_mask.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch: current_seg_mask batch={current_seg_mask.shape[0]}, expected {batch_size}"
        )
    return current_seg_mask.float().clamp(0, 1)


def prepare_future_masks(
    future_seg_masks: Optional[torch.Tensor], batch_size: int
) -> Optional[torch.Tensor]:
    """Normalize future masks to [B, 1, T, H, W]."""
    if future_seg_masks is None:
        return None
    if future_seg_masks.dim() == 4:
        future_seg_masks = future_seg_masks.unsqueeze(1)
    elif future_seg_masks.dim() == 5:
        if future_seg_masks.shape[2] == 1:
            future_seg_masks = future_seg_masks.permute(0, 2, 1, 3, 4)
    if future_seg_masks.shape[0] != batch_size:
        raise ValueError(
            f"Batch mismatch: future_seg_masks batch={future_seg_masks.shape[0]}, expected {batch_size}"
        )
    return future_seg_masks.float().clamp(0, 1)


# ── WAN backbone with mask support ───────────────────────────────────────────

class MaskWanBackbone(nn.Module):
    """
    WAN backbone with mask-conditioned input (Aether-style channel layout).

    The mask stream is concatenated into patch_embedding (48 → 96 ch) whenever
    *either* we want mask-as-input (`input_mask=True`) *or* mask-as-target
    (`predict_mask=True`). This is required for flow matching to see x_t^mask
    when predicting mask velocity.

    `input_mask` now only controls the *source* of the first-frame mask
    condition (real encoded mask vs. zero pad), not the channel count.
    """

    def __init__(self, config: MaskWanMotConfig, predict_mask: bool = False):
        super().__init__()
        self.use_mask = bool(getattr(config, "use_mask", True))
        # input_mask: Aether-style condition source flag.
        #   True  → condition_mask_latent = encode(current_seg_mask)
        #   False → condition_mask_latent = zero pad (no mask signal on input)
        # Does NOT control DiT channel count anymore; that is decided by
        # (input_mask OR predict_mask).
        self.input_mask = bool(getattr(config, "input_mask", self.use_mask))
        self.predict_mask = bool(predict_mask)

        # Read WAN paths from config (set by train script from Hydra cfg.model.wan.*)
        wan_model_path  = getattr(config, "wan_checkpoint_path", "pretrained_models/Wan2.2-TI2V-5B")
        wan_vae_path    = getattr(config, "wan_vae_path", "pretrained_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth")
        wan_config_path = getattr(config, "wan_config_path", "pretrained_models/Wan2.2-TI2V-5B")
        wan_precision   = getattr(config, "wan_precision", "bfloat16")

        lat_T = 1 + config.num_video_frames // 4
        lat_H = config.video_height // 32
        lat_W = config.video_width  // 32
        self.grid_sizes = torch.tensor(
            [lat_T, lat_H, lat_W], dtype=torch.long, device="cuda"
        ).unsqueeze(0).expand(config.batch_size, -1)  # [B, 3]
        print(f"MaskWanBackbone grid_sizes: T={lat_T}, H={lat_H}, W={lat_W}")

        _video_model = WanVideoModel.from_pretrained(
            checkpoint_path=wan_model_path,
            vae_path=wan_vae_path,
            config_path=wan_config_path,
            precision=wan_precision,
            device="cuda",
        )
        self.video_module = VideoModule(_video_model, "bfloat16", "cuda", self.grid_sizes)

        # Record original video latent channels (48 for WAN 2.2)
        self.video_latent_channels = self.video_module.video_model.wan_model.patch_embedding.in_channels  # 48

        # Expand patch_embedding: 48 → 96 (video 48 + mask 48) whenever the
        # mask stream needs to be in DiT input. This is true when either:
        #   - input_mask  (we want to condition on a mask latent), or
        #   - predict_mask (mask is a flow-matching target, so x_t^mask must
        #                   be in the DiT input or the velocity is unlearnable).
        if self.input_mask or self.predict_mask:
            self._expand_patch_embedding(extra_channels=self.video_latent_channels)

        # Optionally expand output head: 48 → 96 when predict_mask=True
        if self.predict_mask:
            self._expand_output_head(extra_out_channels=self.video_latent_channels)

        logger.info(
            "MaskWanBackbone: input_mask=%s, predict_mask=%s, "
            "patch_embedding %d → %d channels, head out_dim %d",
            self.input_mask,
            self.predict_mask,
            self.video_latent_channels,
            self.video_module.video_model.wan_model.patch_embedding.in_channels,
            self.video_module.video_model.wan_model.out_dim,
        )

    def _expand_patch_embedding(self, extra_channels: int) -> None:
        """Expand WAN patch_embedding input channels and zero-init new weights."""
        if extra_channels <= 0:
            return
        old_pe = self.video_module.video_model.wan_model.patch_embedding
        new_in = old_pe.in_channels + extra_channels
        new_pe = nn.Conv3d(
            in_channels=new_in, out_channels=old_pe.out_channels,
            kernel_size=old_pe.kernel_size, stride=old_pe.stride,
            padding=old_pe.padding, dilation=old_pe.dilation,
            groups=old_pe.groups, bias=old_pe.bias is not None,
            padding_mode=old_pe.padding_mode,
            device=old_pe.weight.device, dtype=old_pe.weight.dtype,
        )
        with torch.no_grad():
            new_pe.weight.zero_()
            new_pe.weight[:, :old_pe.in_channels].copy_(old_pe.weight)
            if old_pe.bias is not None:
                new_pe.bias.copy_(old_pe.bias)
        self.video_module.video_model.wan_model.patch_embedding = new_pe
        self.video_module.video_model.wan_model.in_dim = new_in

    def _expand_output_head(self, extra_out_channels: int) -> None:
        """Expand WAN output head to output extra channels, zero-init new weights."""
        if extra_out_channels <= 0:
            return
        old_head = self.video_module.video_model.wan_model.head
        old_out_dim = old_head.out_dim
        new_out_dim = old_out_dim + extra_out_channels
        patch_size = old_head.patch_size

        new_head = Head(
            dim=old_head.dim,
            out_dim=new_out_dim,
            patch_size=patch_size,
            eps=old_head.eps,
        )
        new_head = new_head.to(
            device=old_head.head.weight.device,
            dtype=old_head.head.weight.dtype,
        )

        with torch.no_grad():
            new_head.head.weight.zero_()
            new_head.head.bias.zero_()

            # Head rows are laid out as [patch_t, patch_h, patch_w, channel].
            # Copy video channels at every patch position; leave mask channels zero.
            old_weight = old_head.head.weight.view(
                *patch_size, old_out_dim, old_head.head.in_features
            )
            new_weight = new_head.head.weight.view(
                *patch_size, new_out_dim, new_head.head.in_features
            )
            new_weight[..., :old_out_dim, :].copy_(old_weight)

            old_bias = old_head.head.bias.view(*patch_size, old_out_dim)
            new_bias = new_head.head.bias.view(*patch_size, new_out_dim)
            new_bias[..., :old_out_dim].copy_(old_bias)
            new_head.norm.load_state_dict(old_head.norm.state_dict())
            new_head.modulation.data.copy_(old_head.modulation.data)

        self.video_module.video_model.wan_model.head = new_head
        self.video_module.video_model.wan_model.out_dim = new_out_dim


# ── Policy wrapper ────────────────────────────────────────────────────────────

class MaskDitVideoPolicy(PreTrainedPolicy):
    """
    Wrapper around MaskDitVideoFlowMatching for lerobot training / inference.

    Extends DitVideoPolicy with mask input (always) and optional mask prediction.
    """

    config_class = MaskWanMotConfig
    name = "mask_wan_mot_joint"

    def __init__(
        self,
        config: MaskWanMotConfig,
        dataset_stats: dict | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.model  = MaskDitVideoFlowMatching(config)
        self.reset()

    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self):
        return self.parameters()

    @torch.no_grad()
    def select_action(
        self,
        batch,
        noise=None,
        tau_v=1.0,
        save_video_latent=False,
        use_video_kv_cache: bool = True,
    ):
        self.eval()
        if len(self._action_queue) == 0:
            state = self.prepare_state(batch)
            video_latent_tuple, mask_latent_tuple, language_embeddings = (
                self.prepare_video_mask_latent_and_t5_embeddings(batch)
            )
            results = self.model.sample_actions(
                state, video_latent_tuple, mask_latent_tuple, language_embeddings,
                noise=noise,
                tau_v=tau_v,
                save_video_latent=save_video_latent,
                use_video_kv_cache=use_video_kv_cache,
            )
            actions = results["pred"]
            orig_dim = self.config.action_feature.shape[0]
            actions  = actions[:, :, :orig_dim]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def forward_evaluate(
        self,
        batch,
        noise=None,
        tau_v: float = 0.9,
        save_video_latent: bool = True,
        use_video_kv_cache: bool = True,
    ) -> dict:
        self.eval()
        state     = self.prepare_state(batch)
        actions   = self.prepare_action(batch)
        video_latent_tuple, mask_latent_tuple, language_embeddings = (
            self.prepare_video_mask_latent_and_t5_embeddings(batch)
        )

        results = self.model.sample_actions(
            state, video_latent_tuple, mask_latent_tuple, language_embeddings,
            noise=noise,
            tau_v=tau_v,
            save_video_latent=save_video_latent,
            use_video_kv_cache=use_video_kv_cache,
        )
        orig_dim = self.config.action_feature.shape[0]
        pred = results["pred"][:, :, :orig_dim]
        gt   = actions[:, :, :orig_dim]

        info = {"pred": pred, "gt": gt}
        if "video_latent" in results:
            info["video_latent"] = results["video_latent"]
        if "predicted_masks" in results:
            info["predicted_masks"] = results["predicted_masks"]
        return info

    def forward(self, batch, noise=None, time=None):
        video_latent_tuple, mask_latent_tuple, language_embeddings = (
            self.prepare_video_mask_latent_and_t5_embeddings(batch)
        )
        state   = self.prepare_state(batch)  if batch.get(OBS_STATE) is not None else None
        actions = self.prepare_action(batch) if batch.get(ACTION)    is not None else None
        actions_is_pad = batch.get("actions_is_pad")

        model_out = self.model.forward(
            state, actions, noise, time,
            video_latent_tuple, mask_latent_tuple, language_embeddings,
        )

        if model_out is None:
            return {"loss": torch.tensor(0.0)}

        loss_dict = {}
        if isinstance(model_out, dict):
            losses_flow = model_out["losses_flow"]
            if actions_is_pad is not None:
                in_episode = ~actions_is_pad
                losses_flow = losses_flow * in_episode.unsqueeze(-1)
            losses_flow = losses_flow[:, :, : self.config.max_action_dim]
            flow_loss = losses_flow.mean()

            video_loss = model_out["video_loss"]
            action_w = getattr(self.model, "action_loss_weight", 1.0)
            video_w = getattr(self.model, "video_loss_weight", 1.0)
            total_loss = action_w * flow_loss + video_w * video_loss

            loss_dict["loss"] = total_loss
            loss_dict["flow_loss"] = flow_loss.item()
            loss_dict["video_loss"] = video_loss.item()
            loss_dict["action_loss"] = flow_loss.item()

            # Optional mask loss
            if "mask_loss" in model_out:
                mask_loss = model_out["mask_loss"]
                mask_w = getattr(self.model, "mask_loss_weight", 0.4)
                total_loss = total_loss + mask_w * mask_loss
                loss_dict["loss"] = total_loss
                loss_dict["mask_loss"] = mask_loss.item()
        else:
            losses_flow = model_out
            if actions_is_pad is not None:
                in_episode = ~actions_is_pad
                losses_flow = losses_flow * in_episode.unsqueeze(-1)
            losses_flow = losses_flow[:, :, : self.config.max_action_dim]
            loss_flow = losses_flow.mean()
            loss_dict["flow_loss"] = loss_flow.item()
            loss_dict["loss"]      = loss_flow
        return loss_dict

    # ── Batch preparation ─────────────────────────────────────────────────

    def prepare_state(self, batch):
        return pad_vector(batch[OBS_STATE], self.config.max_state_dim)

    def prepare_action(self, batch):
        return pad_vector(batch[ACTION], self.config.max_action_dim)

    def prepare_video_mask_latent_and_t5_embeddings(self, batch):
        """Prepare video latent, mask latent, and T5 embeddings from batch.

        Returns:
            video_latent_tuple: (clean_full_latent, condition_frame_latent)
            mask_latent_tuple: (clean_full_mask_latent, condition_mask_latent)
            language_embeddings: list of T5 embeddings
        """
        first_frame    = batch["current_image"]
        device         = first_frame.device
        dtype          = first_frame.dtype
        first_frame_n  = (first_frame * 2.0 - 1.0).unsqueeze(2)  # [B, C, 1, H, W]
        B = first_frame.shape[0]

        # ── Video latent encoding ──
        if batch.get("future_images") is not None:
            video_frames = batch["future_images"]
            video_norm   = (video_frames * 2.0 - 1.0).permute(0, 2, 1, 3, 4)
            full_video   = torch.cat([first_frame_n, video_norm], dim=2)
            with torch.no_grad():
                clean_full_latent      = self.model.wan_backbone.video_module.video_model.encode_video(full_video.to(dtype))
                condition_frame_latent = self.model.wan_backbone.video_module.video_model.encode_video(first_frame_n.to(dtype))
        else:
            with torch.no_grad():
                condition_frame_latent = self.model.wan_backbone.video_module.video_model.encode_video(first_frame_n.to(dtype))
            clean_full_latent = None

        # ── Mask latent encoding (Aether-style) ──
        # Mask stream is in DiT input whenever (input_mask OR predict_mask):
        #   condition_mask_latent source:
        #     - input_mask=True  → encode(current_seg_mask) (real mask condition)
        #     - input_mask=False → zero pad (Aether-style "no mask signal" condition)
        # clean_full_mask_latent (flow-matching target) is only needed when
        # predict_mask=True and requires future_seg_masks to be available.
        input_mask   = getattr(self.model, "input_mask", getattr(self.model, "use_mask", True))
        predict_mask = getattr(self.model, "predict_mask", False)

        condition_mask_latent  = None
        clean_full_mask_latent = None

        if input_mask or predict_mask:
            # condition_mask_latent: always present when mask stream is in DiT input.
            if input_mask:
                current_seg_mask = batch.get("current_seg_mask")

                # Modality masking: randomly drop current mask during training for robustness.
                # Replace with all-zero mask (→ black RGB image after mask_to_red_rgb) instead
                # of None, so it goes through the normal VAE encode path.
                if self.training and current_seg_mask is not None:
                    mask_keep_prob = getattr(self.model, "mask_random", 1.0)
                    if torch.rand(1).item() > mask_keep_prob:
                        current_seg_mask = torch.zeros_like(current_seg_mask)

                current_mask = prepare_current_mask(current_seg_mask, B)

                if current_mask is not None:
                    mask_single = current_mask.to(device=device, dtype=dtype).unsqueeze(2)
                    mask_rgb = mask_to_red_rgb(mask_single)
                    with torch.no_grad():
                        condition_mask_latent = self.model.wan_backbone.video_module.video_model.encode_video(
                            mask_rgb.to(dtype)
                        )
                else:
                    condition_mask_latent = torch.zeros_like(condition_frame_latent)
                    logger.warning("input_mask=True but no current_seg_mask in batch; zero-padded.")
            else:
                # Aether-style: no mask signal on input side → zero pad.
                # Skip VAE encoding entirely to save compute.
                condition_mask_latent = torch.zeros_like(condition_frame_latent)

            # clean_full_mask_latent ← supervision target (only when predict_mask)
            if predict_mask:
                current_seg_mask = batch.get("current_seg_mask")
                future_seg_masks = batch.get("future_seg_masks")
                current_mask_t   = prepare_current_mask(current_seg_mask, B)
                future_masks_t   = prepare_future_masks(future_seg_masks, B)
                if current_mask_t is not None and future_masks_t is not None:
                    full_mask = torch.cat([
                        current_mask_t.to(device=device, dtype=dtype).unsqueeze(2),
                        future_masks_t.to(device=device, dtype=dtype),
                    ], dim=2)  # [B, 1, 1+T, H, W]
                    full_mask_rgb = mask_to_red_rgb(full_mask)
                    with torch.no_grad():
                        clean_full_mask_latent = self.model.wan_backbone.video_module.video_model.encode_video(
                            full_mask_rgb.to(dtype)
                        )

        # ── T5 embeddings ──
        t5_embed_batch = batch.get("t5_embed")
        if t5_embed_batch is not None and all(e is not None for e in t5_embed_batch):
            encoded_texts = [emb.to(device=device, dtype=dtype) for emb in t5_embed_batch]
        else:
            raise ValueError(
                "T5 embeddings not found in batch. Ensure the dataset provides 't5_embed'."
            )

        video_latent_tuple = (clean_full_latent, condition_frame_latent)
        mask_latent_tuple  = (clean_full_mask_latent, condition_mask_latent)
        return video_latent_tuple, mask_latent_tuple, encoded_texts


# ── Core model ────────────────────────────────────────────────────────────────

class MaskDitVideoFlowMatching(nn.Module):
    """
    DiT action expert with WAN video-latent conditioning + mask input.

    Extends DitVideoFlowMatching with:
      - Always: mask input (current mask → red RGB → VAE → 48ch, concat with video → 96ch)
      - Optional: mask output prediction (predict_mask=True → head 96ch, split → mask loss)

    Trainable components:
        wan_backbone.video_module.wan_model : WAN transformer backbone
        action_expert              : DitActionExpert
        action_in_proj             : Linear(max_action_dim → proj_width)
        action_out_proj            : Linear(proj_width     → max_action_dim)
        state_proj                 : Linear(max_state_dim  → proj_width)
    """

    def __init__(self, config: MaskWanMotConfig):
        super().__init__()
        self.config = config
        self.action_loss_weight = float(getattr(config, "action_loss_weight", 1.0))
        self.video_loss_weight = float(getattr(config, "video_loss_weight", 1.0))
        self.mask_loss_weight = float(getattr(config, "mask_loss_weight", 1.0))
        self.use_mask = bool(getattr(config, "use_mask", True))
        # input_mask: whether mask is concatenated to DiT input. Defaults to use_mask
        # for backward compat. Independent of predict_mask.
        self.input_mask = bool(getattr(config, "input_mask", self.use_mask))
        self.predict_mask = bool(getattr(config, "predict_mask", False))
        self.mask_random = float(getattr(config, "mask_random", 1.0))

        # ── Trainable WAN backbone with mask support ─────────────────────
        self.wan_backbone = MaskWanBackbone(config, predict_mask=self.predict_mask)

        # Record video latent channels for splitting output
        self.video_latent_channels = self.wan_backbone.video_latent_channels  # 48

        wan_num_layers = self.wan_backbone.video_module.video_model.wan_model.num_layers

        # ── DiT action expert (trainable, reused from wan_mot) ───────────
        self.action_expert = DitActionExpert(
            hidden_size=config.proj_width,          # 1024
            joint_num_heads=24,
            ffn_dim=4096,
            num_layers=wan_num_layers,
            video_hidden_size=3072,                 # WAN 5B hidden dim
            freq_dim=256,
            cross_attn_norm=True,
        )

        # ── Projection heads (trainable) ─────────────────────────────────
        self.action_in_proj  = nn.Linear(config.max_action_dim, config.proj_width)
        self.action_out_proj = nn.Linear(config.proj_width, config.max_action_dim)
        self.state_proj      = nn.Linear(config.max_state_dim,  config.proj_width)

        # ── Flow-matching scheduler (video) ──────────────────────────────
        self.fm_train_scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000
        )
        self.fm_train_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        # ── Flow-matching scheduler (action) ─────────────────────────────
        action_snr_shift = getattr(config, 'action_snr_shift', 1.0)
        self.fm_action_scheduler = FlowMatchScheduler(
            shift=action_snr_shift, sigma_min=0.0, extra_one_step=True, num_train_timesteps=1000
        )
        self.fm_action_scheduler.set_timesteps(num_inference_steps=1000, training=True)

        logger.info(
            "MaskDitVideoFlowMatching: input_mask=%s, predict_mask=%s, "
            "use_mask(legacy)=%s, mask_loss_weight=%.2f",
            self.input_mask, self.predict_mask, self.use_mask, self.mask_loss_weight,
        )

    def sample_noise(self, shape, device):
        return torch.normal(0.0, 1.0, size=shape, dtype=torch.float32, device=device)

    # ── Weight loading ────────────────────────────────────────────────────

    def load_wan_weights(self, checkpoint_path: str) -> None:
        """Load WAN weights – handles patch_embedding (48→96) and head shape mismatch."""
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            safe_load_file = None

        checkpoint_path = os.path.abspath(checkpoint_path)
        logger.info(f"MaskDitVideoFlowMatching: Loading WAN weights from {checkpoint_path}")
        try:
            wan_state_dict = None
            if checkpoint_path.endswith('.pt'):
                loaded = torch.load(checkpoint_path, map_location='cpu')
                wan_state_dict = loaded.get('model', loaded) if isinstance(loaded, dict) and 'model' in loaded else loaded
            elif checkpoint_path.endswith('.safetensors'):
                if safe_load_file is None:
                    raise RuntimeError("safetensors not installed")
                wan_state_dict = safe_load_file(checkpoint_path, device='cpu')
            elif os.path.isdir(checkpoint_path):
                wan_pt = os.path.join(checkpoint_path, 'wan_model.pt')
                safetensors_file = os.path.join(checkpoint_path, 'model.safetensors')
                diffusion_index_json = os.path.join(checkpoint_path, 'diffusion_pytorch_model.safetensors.index.json')
                model_index_json = os.path.join(checkpoint_path, 'model.safetensors.index.json')
                if os.path.exists(wan_pt):
                    loaded = torch.load(wan_pt, map_location='cpu', weights_only=True)
                    wan_state_dict = loaded.get('model', loaded) if isinstance(loaded, dict) and 'model' in loaded else loaded
                elif os.path.exists(safetensors_file):
                    if safe_load_file is None:
                        raise RuntimeError("safetensors not installed")
                    wan_state_dict = safe_load_file(safetensors_file, device='cpu')
                elif os.path.exists(diffusion_index_json) or os.path.exists(model_index_json):
                    if safe_load_file is None:
                        raise RuntimeError("safetensors not installed")
                    sharded_index_file = diffusion_index_json if os.path.exists(diffusion_index_json) else model_index_json
                    import json as _json
                    with open(sharded_index_file, 'r') as f:
                        index_data = _json.load(f)
                    weight_map = index_data.get("weight_map", {})
                    shard_files = sorted(set(weight_map.values()))
                    wan_state_dict = {}
                    for shard_name in shard_files:
                        shard_path = os.path.join(checkpoint_path, shard_name)
                        if not os.path.exists(shard_path):
                            continue
                        shard_dict = safe_load_file(shard_path, device='cpu')
                        wan_state_dict.update(shard_dict)
                else:
                    logger.warning(f"No compatible weight file found in {checkpoint_path}")
                    return

            if wan_state_dict is not None:
                wan_state_dict = _strip_known_prefixes_for_wan(
                    wan_state_dict, self.wan_backbone.video_module.video_model.wan_model
                )

                # Handle patch_embedding shape mismatch (src 48ch → tgt 96ch)
                pe_key = 'patch_embedding.weight'
                if pe_key in wan_state_dict:
                    src_pe = wan_state_dict[pe_key]
                    tgt_pe = self.wan_backbone.video_module.video_model.wan_model.patch_embedding.weight
                    if src_pe.shape != tgt_pe.shape and src_pe.shape[1] < tgt_pe.shape[1]:
                        with torch.no_grad():
                            tgt_pe.data[:, :src_pe.shape[1]] = src_pe.to(tgt_pe.dtype)
                        del wan_state_dict[pe_key]
                    elif src_pe.shape != tgt_pe.shape:
                        del wan_state_dict[pe_key]
                pe_bias_key = 'patch_embedding.bias'
                if pe_bias_key in wan_state_dict:
                    src_bias = wan_state_dict[pe_bias_key]
                    tgt_bias = self.wan_backbone.video_module.video_model.wan_model.patch_embedding.bias
                    if tgt_bias is not None and src_bias.shape != tgt_bias.shape:
                        del wan_state_dict[pe_bias_key]

                # Handle head shape mismatch (src out_dim=48 → tgt out_dim=96 when predict_mask)
                head_weight_key = 'head.head.weight'
                head_bias_key = 'head.head.bias'
                if head_weight_key in wan_state_dict:
                    src_hw = wan_state_dict[head_weight_key]
                    tgt_hw = self.wan_backbone.video_module.video_model.wan_model.head.head.weight
                    if src_hw.shape != tgt_hw.shape and src_hw.shape[0] < tgt_hw.shape[0]:
                        with torch.no_grad():
                            head = self.wan_backbone.video_module.video_model.wan_model.head
                            patch_size = head.patch_size
                            patch_volume = math.prod(patch_size)
                            src_out_dim = src_hw.shape[0] // patch_volume
                            tgt_out_dim = tgt_hw.shape[0] // patch_volume
                            src_weight = src_hw.view(
                                *patch_size, src_out_dim, src_hw.shape[1]
                            )
                            tgt_weight = tgt_hw.view(
                                *patch_size, tgt_out_dim, tgt_hw.shape[1]
                            )
                            tgt_weight[..., :src_out_dim, :].copy_(
                                src_weight.to(tgt_hw.dtype)
                            )
                        del wan_state_dict[head_weight_key]
                    elif src_hw.shape != tgt_hw.shape:
                        del wan_state_dict[head_weight_key]
                if head_bias_key in wan_state_dict:
                    src_hb = wan_state_dict[head_bias_key]
                    tgt_hb = self.wan_backbone.video_module.video_model.wan_model.head.head.bias
                    if src_hb.shape != tgt_hb.shape and src_hb.shape[0] < tgt_hb.shape[0]:
                        with torch.no_grad():
                            head = self.wan_backbone.video_module.video_model.wan_model.head
                            patch_size = head.patch_size
                            patch_volume = math.prod(patch_size)
                            src_out_dim = src_hb.shape[0] // patch_volume
                            tgt_out_dim = tgt_hb.shape[0] // patch_volume
                            src_bias = src_hb.view(*patch_size, src_out_dim)
                            tgt_bias = tgt_hb.view(*patch_size, tgt_out_dim)
                            tgt_bias[..., :src_out_dim].copy_(
                                src_bias.to(tgt_hb.dtype)
                            )
                        del wan_state_dict[head_bias_key]
                    elif src_hb.shape != tgt_hb.shape:
                        del wan_state_dict[head_bias_key]

                incompatible = self.wan_backbone.video_module.video_model.wan_model.load_state_dict(
                    wan_state_dict, strict=False
                )
                if incompatible.missing_keys:
                    logger.warning(f"Missing keys: {len(incompatible.missing_keys)}")
                if incompatible.unexpected_keys:
                    logger.warning(f"Unexpected keys: {len(incompatible.unexpected_keys)}")
                logger.info("Successfully loaded WAN weights")
        except Exception as e:
            logger.error(f"Failed to load WAN weights: {e}")
            logger.warning("Using random initialization for WAN model")

    # ── Embedding helpers ─────────────────────────────────────────────────

    def _embed_mid(self, noisy_latent_input, timestep_per_token):
        """Embed the latent input (96ch when use_mask=True, 48ch otherwise) through patch_embedding."""
        video_emb = self.wan_backbone.video_module.prepare_input(
            noisy_latent_input.to(torch.bfloat16)
        )
        time_emb, video_adaln_params = self.wan_backbone.video_module.get_time_embedding(
            timestep_per_token, video_emb.shape[1]
        )
        video_adaln_params = video_adaln_params.to(torch.bfloat16)

        bsize, video_token_num = video_emb.shape[:2]
        device     = noisy_latent_input.device
        video_mask = torch.ones(bsize, video_token_num, dtype=torch.bool, device=device)
        att_masks  = [1] + [0] * (video_token_num - 1)
        att_masks_t = torch.tensor(att_masks, dtype=video_emb.dtype, device=device)[None].expand(bsize, -1)

        return video_emb, video_mask, att_masks_t, video_adaln_params, time_emb

    def _build_masked_joint_attention(
        self,
        video_q: torch.Tensor | None,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_q: torch.Tensor | None,
        action_k: torch.Tensor | None,
        action_v: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Shared masked attention kernel for train and infer.

        Uses flash_attn_func explicitly to guarantee Flash Attention 2 kernel.
        Mask logic (video cannot see action, action sees all) is implemented by
        splitting into two separate flash_attn calls — both mask-free.
        """
        head_dim = video_k.shape[3]
        softmax_scale = head_dim ** -0.5
        video_len = video_k.shape[1]
        action_len = 0 if action_q is None else action_q.shape[1]

        # All tensors are [B, S, H, D] — exactly what flash_attn_func expects
        video_attn_out_h = None
        action_attn_out_h = None

        if video_q is not None and action_len > 0:
            # Case: both video and action present (training)
            # Video self-attn: video_q attends only to video_kv
            video_attn_out_h = _fa2(
                video_q, video_k, video_v, softmax_scale=softmax_scale,
            )  # [B, L_v, H, D]

            # Action cross-attn: action_q attends to all KV (video + action)
            all_k = torch.cat([video_k, action_k], dim=1) if action_k is not None else video_k
            all_v = torch.cat([video_v, action_v], dim=1) if action_v is not None else video_v
            action_attn_out_h = _fa2(
                action_q, all_k, all_v, softmax_scale=softmax_scale,
            )  # [B, L_a, H, D]

        elif video_q is not None:
            # Case: video only (no action) — pure self-attn
            video_attn_out_h = _fa2(
                video_q, video_k, video_v, softmax_scale=softmax_scale,
            )

        elif action_q is not None:
            # Case: action only (inference action denoising)
            all_k = torch.cat([video_k, action_k], dim=1) if action_k is not None else video_k
            all_v = torch.cat([video_v, action_v], dim=1) if action_v is not None else video_v
            action_attn_out_h = _fa2(
                action_q, all_k, all_v, softmax_scale=softmax_scale,
            )

        return video_attn_out_h, action_attn_out_h

    def _compute_video_qkv(
        self,
        video_tokens: torch.Tensor,
        video_adaln_mod: tuple,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        wan_model = self.wan_backbone.video_module.video_model.wan_model
        wan_layer = wan_model.blocks[layer_idx]
        wan_attn = wan_layer.self_attn
        batch_size, video_len, _ = video_tokens.shape

        norm_video = wan_layer.norm1(video_tokens).float()
        norm_video = norm_video * (1 + video_adaln_mod[1].squeeze(2)) + video_adaln_mod[0].squeeze(2)

        video_q = wan_attn.norm_q(wan_attn.q(norm_video)).view(batch_size, video_len, wan_attn.num_heads, wan_attn.head_dim)
        video_k = wan_attn.norm_k(wan_attn.k(norm_video)).view(batch_size, video_len, wan_attn.num_heads, wan_attn.head_dim)
        video_v = wan_attn.v(norm_video).view(batch_size, video_len, wan_attn.num_heads, wan_attn.head_dim)

        freqs = wan_model.freqs.to(video_tokens.device)
        grid_sizes = self.wan_backbone.grid_sizes[:batch_size].to(video_tokens.device)
        video_q = rope_apply(video_q, grid_sizes, freqs)
        video_k = rope_apply(video_k, grid_sizes, freqs)
        return video_q.to(video_tokens.dtype), video_k.to(video_tokens.dtype), video_v.to(video_tokens.dtype), wan_attn

    def _run_wan_joint_layer(
        self,
        layer_idx: int,
        video_adaln_mod: tuple,
        video_tokens: torch.Tensor | None = None,
        action_tokens: torch.Tensor | None = None,
        timestep_proj: torch.Tensor | None = None,
        kv_cache: dict | None = None,
        fill_kv_cache: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, tuple | None, dict | None]:
        """Unified masked joint-attention layer (same as wan_mot)."""
        cache_entry = None
        action_adaln = None
        video_k = video_v = video_q = wan_attn = None

        if video_tokens is not None:
            video_q, video_k, video_v, wan_attn = self._compute_video_qkv(
                video_tokens, video_adaln_mod, layer_idx=layer_idx,
            )
            if fill_kv_cache:
                cache_entry = {"k": video_k, "v": video_v}
        elif kv_cache is not None:
            video_k = kv_cache["k"]
            video_v = kv_cache["v"]
        else:
            raise ValueError("Either video_tokens or kv_cache must be provided.")

        action_q = action_k = action_v = None
        if action_tokens is not None:
            if timestep_proj is None:
                raise ValueError("timestep_proj is required when action_tokens is provided.")
            action_q, action_k, action_v, action_adaln = self.action_expert.prepare_wan_joint_qkv(
                action_tokens, timestep_proj, layer_idx=layer_idx,
                num_heads=video_k.shape[2], head_dim=video_k.shape[3],
            )

        video_attn_out_h, action_attn_out_h = self._build_masked_joint_attention(
            video_q, video_k, video_v, action_q, action_k, action_v,
        )

        if video_tokens is not None:
            if wan_attn is None:
                _, _, _, wan_attn = self._compute_video_qkv(video_tokens, video_adaln_mod, layer_idx)
            video_attn_out = wan_attn.o(video_attn_out_h.flatten(2).to(video_tokens.dtype))
            video_tokens = video_tokens + video_attn_out * video_adaln_mod[2].squeeze(2)

        if action_tokens is not None:
            action_tokens = self.action_expert.apply_wan_joint_output(
                action_tokens, action_attn_out_h, action_adaln, layer_idx=layer_idx,
            )

        return video_tokens, action_tokens, action_adaln, cache_entry

    def _run_wan_video_only(
        self,
        noisy_latent_input: torch.Tensor,
        timestep_per_token: torch.Tensor,
        processed_t5: torch.Tensor,
        need_video_pred: bool = False,
        fill_kv_cache: bool = False,
    ) -> tuple[torch.Tensor | None, list[dict] | None]:
        """Run WAN alone (96ch input when use_mask=True, 48ch when use_mask=False)."""
        mid_embs, _, _, video_adaln_params, time_emb = self._embed_mid(noisy_latent_input, timestep_per_token)
        num_layers = self.wan_backbone.video_module.video_model.wan_model.num_layers
        video_kv_cache = [] if fill_kv_cache else None

        with torch.autocast(device_type="cuda", dtype=noisy_latent_input.dtype):
            for i in range(num_layers):
                video_adaln_mod = self.wan_backbone.video_module.compute_adaln_modulation(
                    video_adaln_params, i
                )
                mid_embs, _, _, cache_entry = self._run_wan_joint_layer(
                    layer_idx=i,
                    video_adaln_mod=video_adaln_mod,
                    video_tokens=mid_embs,
                    action_tokens=None,
                    timestep_proj=None,
                    kv_cache=None,
                    fill_kv_cache=fill_kv_cache,
                )
                mid_embs = self.wan_backbone.video_module.process_cross_attention(
                    mid_embs, video_adaln_params, i, processed_t5
                )
                mid_embs = self.wan_backbone.video_module.process_ffn(mid_embs, video_adaln_mod, i)
                if fill_kv_cache:
                    video_kv_cache.append(cache_entry)

            video_pred = None
            if need_video_pred:
                video_pred = self.wan_backbone.video_module.apply_output_head(mid_embs, time_emb)

        return video_pred, video_kv_cache

    def _run_action_with_video_cache(
        self,
        action_embs: torch.Tensor,
        action_time: torch.Tensor,
        video_kv_cache: list[dict],
    ) -> torch.Tensor:
        """Run action denoising with cached video KV (same as wan_mot)."""
        timestep_proj = self.action_expert.compute_timestep_proj(
            action_time, dtype=action_embs.dtype,
        )
        action_tokens = action_embs

        for i, cache_entry in enumerate(video_kv_cache):
            _, action_tokens, action_adaln, _ = self._run_wan_joint_layer(
                layer_idx=i,
                video_adaln_mod=None,
                video_tokens=None,
                action_tokens=action_tokens,
                timestep_proj=timestep_proj,
                kv_cache=cache_entry,
                fill_kv_cache=False,
            )
            action_tokens = self.action_expert.apply_ffn(
                action_tokens, action_adaln, layer_idx=i,
            )

        return self.action_expert.finalize(action_tokens, timestep_proj)

    def _run_action_with_joint_video(
        self,
        latent_input: torch.Tensor,
        timestep_per_token: torch.Tensor,
        processed_t5: torch.Tensor,
        action_embs: torch.Tensor,
        action_time: torch.Tensor,
    ) -> torch.Tensor:
        """Inference ablation: recompute the train-like joint path."""
        action_tokens, _ = self._run_wan_action_interleaved(
            latent_input, timestep_per_token, processed_t5,
            action_embs, action_time, need_video_pred=False,
        )
        return action_tokens

    def _run_wan_action_interleaved(
        self,
        noisy_latent_input: torch.Tensor,
        timestep_per_token: torch.Tensor,
        processed_t5,
        action_embs: torch.Tensor,
        action_time: torch.Tensor,
        need_video_pred: bool = False,
    ):
        """Run WAN and action expert block-by-block with joint attention (same as wan_mot)."""
        mid_embs, _, _, video_adaln_params, time_emb = self._embed_mid(noisy_latent_input, timestep_per_token)
        num_layers = self.wan_backbone.video_module.video_model.wan_model.num_layers

        timestep_proj = self.action_expert.compute_timestep_proj(
            action_time, dtype=action_embs.dtype,
        )
        action_tokens = action_embs

        with torch.autocast(device_type="cuda", dtype=noisy_latent_input.dtype):
            for i in range(num_layers):
                video_adaln_mod = self.wan_backbone.video_module.compute_adaln_modulation(
                    video_adaln_params, i
                )
                mid_embs, action_tokens, action_adaln, _ = self._run_wan_joint_layer(
                    layer_idx=i,
                    video_adaln_mod=video_adaln_mod,
                    video_tokens=mid_embs,
                    action_tokens=action_tokens,
                    timestep_proj=timestep_proj,
                    kv_cache=None,
                    fill_kv_cache=False,
                )
                mid_embs = self.wan_backbone.video_module.process_cross_attention(
                    mid_embs, video_adaln_params, i, processed_t5
                )
                mid_embs = self.wan_backbone.video_module.process_ffn(mid_embs, video_adaln_mod, i)
                action_tokens = self.action_expert.apply_ffn(
                    action_tokens, action_adaln, layer_idx=i,
                )

            video_pred = None
            if need_video_pred:
                video_pred = self.wan_backbone.video_module.apply_output_head(mid_embs, time_emb)

        action_tokens = self.action_expert.finalize(action_tokens, timestep_proj)
        return action_tokens, video_pred

    # ── Action embedding ──────────────────────────────────────────────────

    def embed_suffix_dit(
        self,
        state: torch.Tensor,         # [B, max_state_dim]
        noisy_actions: torch.Tensor,  # [B, n_act, max_action_dim]
    ) -> torch.Tensor:
        """Embed state + noisy actions → [B, 1+n_act, proj_width]."""
        state_emb  = self.state_proj(state).to(dtype=torch.bfloat16)
        action_emb = self.action_in_proj(noisy_actions.to(torch.bfloat16))
        return torch.cat([state_emb[:, None, :], action_emb], dim=1)

    # ── Training forward ──────────────────────────────────────────────────

    def forward(
        self,
        state,
        actions,
        noise=None,
        time=None,
        video_latent_tuple=None,
        mask_latent_tuple=None,
        language_embeddings=None,
    ) -> Optional[Tensor | dict]:
        """Flow-matching loss with mask-conditioned video + action.

        Returns:
            dict with per-element action flow loss, scalar video loss,
            and optional scalar mask loss.
        """
        if actions is None:
            return None

        (clean_full_latent, condition_frame_latent) = video_latent_tuple
        (clean_full_mask_latent, condition_mask_latent) = mask_latent_tuple

        if clean_full_latent is None:
            raise ValueError("clean_full_latent is None. Joint training requires `future_images` in batch.")

        B = clean_full_latent.shape[0]
        dtype = condition_frame_latent.dtype
        device = condition_frame_latent.device

        # ── Video flow matching noise ──
        timestep_id = torch.randint(0, self.fm_train_scheduler.num_train_timesteps, (B,))
        time_video = self.fm_train_scheduler.timesteps[timestep_id].to(dtype=dtype, device=device)
        sigma = self.fm_train_scheduler.sigmas[timestep_id].to(dtype=dtype, device=device).view(B, 1, 1, 1, 1)
        video_noise = torch.randn_like(clean_full_latent, dtype=dtype)
        noisy_video = clean_full_latent * (1 - sigma) + video_noise * sigma

        _, mask2 = masks_like([noisy_video[0]], zero=True)
        mask = mask2[0].unsqueeze(0)
        noisy_video = (1.0 - mask) * condition_frame_latent + mask * noisy_video
        video_target = video_noise - clean_full_latent
        video_target[:, :, 0:1] = 0

        # ── Mask flow matching noise (Aether-style, shared sigma) ──
        # Mask stream is concatenated to DiT input whenever (input_mask OR predict_mask).
        # `input_mask` only decides the first-frame condition *source*:
        #   True  → real encoded mask (condition_mask_latent from encoder)
        #   False → zero pad (already set in prepare_video_mask_latent_and_t5_embeddings)
        mask_target = None
        noisy_mask  = None

        if self.input_mask or self.predict_mask:
            assert condition_mask_latent is not None, (
                "condition_mask_latent must be set whenever (input_mask OR predict_mask). "
                "Check prepare_video_mask_latent_and_t5_embeddings."
            )
            if self.predict_mask and clean_full_mask_latent is not None:
                # Flow matching on mask: sample ε_mask, build x_t, target = ε - x_0
                mask_noise = torch.randn_like(clean_full_mask_latent, dtype=dtype)
                diffused_mask = clean_full_mask_latent * (1 - sigma) + mask_noise * sigma
                mask_target = mask_noise - clean_full_mask_latent
                mask_target[:, :, 0:1] = 0
                # First frame clamped to condition (real or zero-pad), rest = diffused.
                noisy_mask = (1.0 - mask) * condition_mask_latent + mask * diffused_mask
            else:
                # input_mask=True, predict_mask=False: static mask condition, no target.
                # First frame = condition, rest = zero (no denoising on mask stream).
                noisy_mask = (1.0 - mask) * condition_mask_latent + mask * torch.zeros_like(condition_mask_latent)

        # ── Concat video + (optional) mask stream ──
        if noisy_mask is not None:
            latent_input = torch.cat([noisy_video, noisy_mask], dim=1)  # [B, 96, T', H', W']
        else:
            latent_input = noisy_video  # [B, 48, T', H', W']  (only when both flags are False)

        patch_size_h, patch_size_w = 2, 2
        mask_template = mask[0, 0][:, ::patch_size_h, ::patch_size_w].flatten()
        timestep_per_token = time_video.unsqueeze(1) * mask_template.unsqueeze(0)

        # ── Action flow matching noise ──
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            action_timestep_id = torch.randint(
                0, self.fm_action_scheduler.num_train_timesteps, (B,)
            )
            action_sigma = self.fm_action_scheduler.sigmas[action_timestep_id].to(dtype=dtype, device=device)
            time = action_sigma
        else:
            action_sigma = time

        action_sigma_3d = action_sigma.view(B, 1, 1)
        x_t = (1.0 - action_sigma_3d) * actions + action_sigma_3d * noise
        u_t = noise - actions

        processed_t5 = self.wan_backbone.video_module.preprocess_t5_embeddings(language_embeddings)
        action_embs = self.embed_suffix_dit(state, x_t)
        action_out, full_pred = self._run_wan_action_interleaved(
            latent_input,
            timestep_per_token,
            processed_t5,
            action_embs,
            time,
            need_video_pred=True,
        )

        # ── Video loss ──
        if self.predict_mask:
            # full_pred: [B, 96, T', H', W'] → split
            video_pred = full_pred[:, :self.video_latent_channels]
            mask_pred  = full_pred[:, self.video_latent_channels:]
        else:
            # full_pred: [B, 48, T', H', W']
            video_pred = full_pred

        sq_error = torch.pow(video_pred - video_target, 2)
        video_loss = (sq_error * mask).sum() / (mask.sum() + 1e-6)

        # ── Mask loss (optional) ──
        result = {}
        if self.predict_mask and mask_target is not None:
            mask_sq_error = torch.pow(mask_pred - mask_target, 2)
            mask_loss = (mask_sq_error * mask).sum() / (mask.sum() + 1e-6)
            result["mask_loss"] = mask_loss

        # ── Action loss ──
        action_out = action_out[:, -self.config.n_action_steps:]
        proj_dtype = self.action_out_proj.weight.dtype
        v_t = self.action_out_proj(action_out.to(proj_dtype))

        losses_flow = F.mse_loss(u_t.float(), v_t.float(), reduction="none")
        result["losses_flow"] = losses_flow
        result["video_loss"] = video_loss
        return result

    # ── Inference ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample_actions(
        self,
        state,
        video_latent_tuple,
        mask_latent_tuple,
        language_embeddings,
        noise=None,
        tau_v: float = 0.9,
        save_video_latent: bool = False,
        use_video_kv_cache: bool = True,
        video_num_inference_steps: int = 50,
    ) -> dict:
        """Denoise video+mask to tau_v, then denoise actions with cached KV."""
        bsize  = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)
        x_t = noise

        # ── 1. Video+Mask denoising loop ──
        num_inference_steps = int(video_num_inference_steps)
        if num_inference_steps < 1:
            raise ValueError("video_num_inference_steps 必须大于 0")
        (_, condition_frame_latent) = video_latent_tuple
        (_, condition_mask_latent)  = mask_latent_tuple
        video_dtype = condition_frame_latent.dtype
        B = condition_frame_latent.shape[0]

        _, C_lat, _, H_lat, W_lat = condition_frame_latent.shape
        num_total_latent_frames    = 1 + self.config.num_video_frames // 4
        video_noise = torch.randn(
            (B, C_lat, num_total_latent_frames, H_lat, W_lat), device=device, dtype=video_dtype
        )

        _, mask2 = masks_like([video_noise[0]], zero=True)
        mask     = mask2[0].unsqueeze(0)
        patch_size_h, patch_size_w = 2, 2
        video_latent = (1.0 - mask) * condition_frame_latent + mask * video_noise

        # Mask latent initialization (Aether-style).
        # Mask stream is in DiT input whenever (input_mask OR predict_mask).
        # `condition_mask_latent` is always present in that case, either:
        #   - a real encoded mask (input_mask=True), or
        #   - a zero pad tensor    (input_mask=False), both set by the policy in
        #     prepare_video_mask_latent_and_t5_embeddings.
        if self.input_mask or self.predict_mask:
            assert condition_mask_latent is not None, (
                "condition_mask_latent must be provided when (input_mask OR predict_mask)."
            )
            mask_noise = torch.randn(
                (B, C_lat, num_total_latent_frames, H_lat, W_lat), device=device, dtype=video_dtype
            )
            if self.predict_mask:
                # Flow-matching inference: start from pure noise, clamp first frame to condition.
                mask_latent = (1.0 - mask) * condition_mask_latent + mask * mask_noise
            else:
                # Static mask condition (no denoising on mask stream): first frame condition, rest zero.
                mask_latent = (1.0 - mask) * condition_mask_latent + mask * torch.zeros_like(mask_noise)
        else:
            mask_latent = None

        sample_scheduler  = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
        )
        sampling_sigmas   = get_sampling_sigmas(num_inference_steps, 5.0)
        timesteps, _      = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        timesteps_filtered = [t for t in timesteps if t >= tau_v * 1000] or [timesteps[1]]

        # Separate scheduler for mask stream when predict_mask
        if self.predict_mask:
            mask_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
            )
            retrieve_timesteps(mask_scheduler, device=device, sigmas=sampling_sigmas)

        processed_t5 = self.wan_backbone.video_module.preprocess_t5_embeddings(language_embeddings)

        for t in timesteps_filtered:
            # Concat video + mask → 96ch whenever mask stream is active (Aether-style).
            if mask_latent is not None:
                latent_input = torch.cat([
                    video_latent.to(video_dtype),
                    mask_latent.to(video_dtype),
                ], dim=1)
            else:
                latent_input = video_latent.to(video_dtype)

            temp_ts = (mask2[0][0][:, ::patch_size_h, ::patch_size_w] * t).flatten()
            timestep_per_tok = temp_ts.unsqueeze(0)
            full_pred, _ = self._run_wan_video_only(
                latent_input, timestep_per_tok, processed_t5,
                need_video_pred=True, fill_kv_cache=False,
            )

            # Output head is 96ch iff predict_mask=True (independent of input_mask)
            if self.predict_mask:
                video_noise_pred = full_pred[:, :C_lat]
                mask_noise_pred  = full_pred[:, C_lat:]
            else:
                video_noise_pred = full_pred

            # Video scheduler step
            temp_x0 = sample_scheduler.step(video_noise_pred, t, video_latent, return_dict=False)[0]
            video_latent = temp_x0
            video_latent = (1.0 - mask) * condition_frame_latent + mask * video_latent

            # Mask scheduler step (only when predicting mask)
            if self.predict_mask:
                mask_x0 = mask_scheduler.step(mask_noise_pred, t, mask_latent, return_dict=False)[0]
                mask_latent = mask_x0
                # Clamp first frame to the provided condition (real or zero pad)
                mask_latent = (1.0 - mask) * condition_mask_latent + mask * mask_latent

        # ── 2. Build video KV cache at tau_v ──
        # Concat for cache building: 96ch whenever mask stream is active.
        if mask_latent is not None:
            cache_latent_input = torch.cat([
                video_latent.to(video_dtype),
                mask_latent.to(video_dtype),
            ], dim=1)
        else:
            cache_latent_input = video_latent.to(video_dtype)

        tau_ts = (mask2[0][0][:, ::patch_size_h, ::patch_size_w] * tau_v * 1000).flatten()
        tau_timestep_per_token = tau_ts.unsqueeze(0)
        video_kv_cache = None
        if use_video_kv_cache:
            _, video_kv_cache = self._run_wan_video_only(
                cache_latent_input, tau_timestep_per_token, processed_t5,
                need_video_pred=False, fill_kv_cache=True,
            )

        # ── 3. Action denoising loop ──
        dt = -1.0 / self.config.num_steps
        dt_t = torch.tensor(dt, dtype=torch.float32, device=device)
        time_a = torch.tensor(1.0, dtype=torch.float32, device=device)

        while time_a >= -dt_t / 2:
            expanded_time = time_a.expand(bsize)
            action_embs = self.embed_suffix_dit(state, x_t)
            if use_video_kv_cache:
                action_out = self._run_action_with_video_cache(
                    action_embs, expanded_time, video_kv_cache,
                )
            else:
                action_out = self._run_action_with_joint_video(
                    cache_latent_input, tau_timestep_per_token, processed_t5,
                    action_embs, expanded_time,
                )
            action_out = action_out[:, -self.config.n_action_steps:]
            proj_dtype = self.action_out_proj.weight.dtype
            v_t = self.action_out_proj(action_out.to(proj_dtype)).float()
            x_t = x_t + dt_t * v_t
            time_a = time_a + dt_t

        # ── 4. Optionally decode and save video/mask frames ──
        video_latent_dict = None
        predicted_masks_out = None
        if save_video_latent:
            videos = self.wan_backbone.video_module.video_model.decode_video(video_latent)
            predicted_frames = videos[:, :, 1:]
            predicted_frames = ((predicted_frames + 1.0) / 2.0).clamp(0, 1).float()

            os.makedirs("debug_videos", exist_ok=True)
            for batch_idx in range(predicted_frames.shape[0]):
                frames_list = [
                    Image.fromarray(
                        (predicted_frames[batch_idx, :, t_idx].permute(1, 2, 0).cpu().numpy() * 255
                         ).astype("uint8")
                    )
                    for t_idx in range(predicted_frames.shape[2])
                ]
                gif_path = f"debug_videos/mask_dit_tau{tau_v:.2f}_b{batch_idx}.gif"
                frames_list[0].save(gif_path, save_all=True, append_images=frames_list[1:],
                                    duration=100, loop=0)
                print(f"Saved {gif_path}")

            video_latent_dict = {
                "noisy_video_latent": video_latent.cpu(),
                "tau_v": tau_v,
                "predicted_frames": predicted_frames.cpu(),
            }

            # Decode predicted masks if predict_mask is enabled, and save them
            # next to the video gifs (mask + side-by-side overlay) for inspection.
            if self.predict_mask and mask_latent is not None:
                mask_rgb_decoded = self.wan_backbone.video_module.video_model.decode_video(mask_latent)
                mask_decoded = mask_rgb_decoded[:, :, 1:]  # Skip first frame
                mask_decoded = ((mask_decoded + 1.0) / 2.0).clamp(0, 1).float()
                predicted_masks_out = (mask_decoded[:, 0:1] > 0.5).float()  # Red channel → binary

                for batch_idx in range(mask_decoded.shape[0]):
                    mask_frames_np = [
                        (mask_decoded[batch_idx, :, t_idx].permute(1, 2, 0).cpu().numpy() * 255
                         ).astype("uint8")
                        for t_idx in range(mask_decoded.shape[2])
                    ]
                    mask_gif_path = (
                        f"debug_videos/mask_dit_tau{tau_v:.2f}_b{batch_idx}_mask.gif"
                    )
                    mask_pil = [Image.fromarray(f) for f in mask_frames_np]
                    mask_pil[0].save(
                        mask_gif_path, save_all=True, append_images=mask_pil[1:],
                        duration=100, loop=0,
                    )
                    print(f"Saved {mask_gif_path}")

                    # Side-by-side video|mask preview (resize mask to video HW if needed).
                    overlay_frames = []
                    vH, vW = predicted_frames.shape[-2:]
                    for t_idx in range(mask_decoded.shape[2]):
                        v_np = (
                            predicted_frames[batch_idx, :, t_idx]
                            .permute(1, 2, 0).cpu().numpy() * 255
                        ).astype("uint8")
                        m_np = mask_frames_np[t_idx]
                        if m_np.shape[:2] != (vH, vW):
                            m_pil = Image.fromarray(m_np).resize(
                                (vW, vH), resample=Image.NEAREST
                            )
                            m_np = np.asarray(m_pil)
                        overlay_frames.append(
                            Image.fromarray(np.hstack([v_np, m_np]))
                        )
                    overlay_gif_path = (
                        f"debug_videos/mask_dit_tau{tau_v:.2f}_b{batch_idx}_overlay.gif"
                    )
                    overlay_frames[0].save(
                        overlay_gif_path, save_all=True,
                        append_images=overlay_frames[1:], duration=100, loop=0,
                    )
                    print(f"Saved {overlay_gif_path}")

        res = {"pred": x_t}
        if save_video_latent:
            res["video_latent"] = video_latent_dict
        if predicted_masks_out is not None:
            res["predicted_masks"] = predicted_masks_out
        return res
