"""Miscellaneous helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def seed_everything(seed: int = 42) -> None:
    """Seed python, numpy and torch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count (trainable) parameters of a module."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)
