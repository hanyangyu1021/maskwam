"""Closed-loop evaluation in simulation benchmarks."""

from __future__ import annotations

import torch.nn as nn


class Evaluator:
    """Rolls out a trained policy in a simulator and reports success rates."""

    def __init__(self, model: nn.Module, cfg: dict) -> None:
        self.model = model
        self.cfg = cfg
        # TODO: connect to the benchmark env (LIBERO / RoboTwin 2.0).

    def evaluate(self) -> dict:
        """Run rollouts and return per-task and aggregate success rates."""
        raise NotImplementedError("Evaluator.evaluate is not released yet.")
