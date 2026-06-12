"""Action prediction head."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActionHead(nn.Module):
    """Predicts a chunk of future actions from fused world-action tokens."""

    def __init__(self, hidden_dim: int = 768, action_dim: int = 7, chunk_size: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        # TODO: build the decoder (e.g. diffusion / regression head).

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map fused tokens to actions ``(B, chunk_size, action_dim)``."""
        raise NotImplementedError("ActionHead.forward is not released yet.")
