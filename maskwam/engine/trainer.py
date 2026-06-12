"""Training loop for MaskWAM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """Minimal trainer that wraps optimization, logging and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        cfg: dict,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.step = 0
        # TODO: build optimizer, lr scheduler, grad scaler, logger.

    def train(self) -> None:
        """Run the full training schedule."""
        raise NotImplementedError("Trainer.train is not released yet.")

    def train_step(self, batch: dict) -> dict:
        """Run a single optimization step and return a dict of scalars."""
        raise NotImplementedError("Trainer.train_step is not released yet.")

    def save_checkpoint(self, path: str) -> None:
        raise NotImplementedError("Trainer.save_checkpoint is not released yet.")
