"""Core World-Action Model that unifies RGB, mask, and action modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from maskwam.models.action_head import ActionHead
from maskwam.models.mask_encoder import MaskEncoder
from maskwam.models.rgb_encoder import RGBEncoder
from maskwam.models.tokenizer import MultiModalTokenizer


@dataclass
class MaskWAMConfig:
    """Configuration for :class:`MaskWAM`."""

    rgb_encoder: str = "vit_base_patch16"
    mask_encoder: str = "vit_small_patch16"
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    action_dim: int = 7
    action_chunk_size: int = 8
    pred_horizon: int = 4
    use_mask_prompt: bool = True
    predict_future_mask: bool = True
    dropout: float = 0.1
    loss_weights: dict = field(
        default_factory=lambda: {"rgb": 1.0, "mask": 1.0, "action": 1.0}
    )


class MaskWAM(nn.Module):
    """Object-centric World-Action Model.

    MaskWAM jointly predicts future RGB frames, future masks, and action chunks.
    Masks are used both as an explicit first-frame spatial prompt (mask prompting)
    and as future prediction targets (mask prediction), within a single unified
    RGB-mask-action transformer backbone.
    """

    def __init__(self, config: MaskWAMConfig) -> None:
        super().__init__()
        self.config = config

        self.rgb_encoder = RGBEncoder(config.rgb_encoder, out_dim=config.hidden_dim)
        self.mask_encoder = MaskEncoder(config.mask_encoder, out_dim=config.hidden_dim)
        self.tokenizer = MultiModalTokenizer(hidden_dim=config.hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.action_head = ActionHead(
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim,
            chunk_size=config.action_chunk_size,
        )
        # TODO: future RGB / mask decoders for world-model prediction.
        self.rgb_decoder: Optional[nn.Module] = None
        self.mask_decoder: Optional[nn.Module] = None

    def forward(
        self,
        rgb: torch.Tensor,
        language: torch.Tensor,
        mask_prompt: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run a forward pass.

        Args:
            rgb: ``(B, T, 3, H, W)`` observation frames.
            language: tokenized instruction, ``(B, L)``.
            mask_prompt: optional first-frame target mask, ``(B, 1, H, W)``.

        Returns:
            Dict with predicted ``actions``, ``future_rgb`` and ``future_mask``.
        """
        # TODO: implement RGB/mask/language tokenization, fusion through the
        # backbone, and the three prediction heads.
        raise NotImplementedError("MaskWAM.forward is not released yet.")

    @torch.no_grad()
    def predict_action(
        self,
        rgb: torch.Tensor,
        language: torch.Tensor,
        mask_prompt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict the next action chunk for closed-loop control."""
        raise NotImplementedError("MaskWAM.predict_action is not released yet.")

    def compute_loss(self, batch: dict) -> dict:
        """Compute the joint RGB + mask + action training loss."""
        raise NotImplementedError("MaskWAM.compute_loss is not released yet.")
