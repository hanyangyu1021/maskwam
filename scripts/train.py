"""Entry point for training MaskWAM.

Usage:
    python scripts/train.py --config configs/train/libero.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MaskWAM.")
    parser.add_argument("--config", required=True, help="Path to a training YAML config.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load config, build dataset/model/trainer, then trainer.train().
    raise NotImplementedError("Training entry point is not released yet.")


if __name__ == "__main__":
    main()
