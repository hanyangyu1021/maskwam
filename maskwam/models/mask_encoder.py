"""Mask encoder used for both mask prompting and mask prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    """Encodes binary/instance masks into latent tokens.

    The same encoder is reused for (1) the optional first-frame mask prompt and
    (2) the future-mask prediction targets used as object-centric supervision.
    """

    def __init__(self, backbone: str = "vit_small_patch16", out_dim: int = 768) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.out_dim = out_dim
        # TODO: instantiate the mask backbone and projection layer.

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, 1, H, W)`` masks to ``(B, T, N, out_dim)`` tokens."""
        raise NotImplementedError("MaskEncoder.forward is not released yet.")
