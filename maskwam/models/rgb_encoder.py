"""RGB observation encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class RGBEncoder(nn.Module):
    """Encodes RGB frames into a sequence of visual latent tokens."""

    def __init__(self, backbone: str = "vit_base_patch16", out_dim: int = 768) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.out_dim = out_dim
        # TODO: instantiate the ViT backbone (e.g. via timm) and projection.

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, 3, H, W)`` frames to ``(B, T, N, out_dim)`` tokens."""
        raise NotImplementedError("RGBEncoder.forward is not released yet.")
