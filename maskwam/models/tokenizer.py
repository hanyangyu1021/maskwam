"""Multi-modal tokenizer that assembles RGB, mask, language and action tokens."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiModalTokenizer(nn.Module):
    """Packs heterogeneous modality tokens into a single transformer sequence.

    Adds modality-type embeddings and temporal/positional embeddings so that the
    shared backbone can attend across RGB, mask, language and action tokens.
    """

    def __init__(self, hidden_dim: int = 768, max_seq_len: int = 2048) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        # TODO: define modality-type embeddings and positional encodings.

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        mask_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate modality tokens into ``(B, S, hidden_dim)``."""
        raise NotImplementedError("MultiModalTokenizer.forward is not released yet.")
