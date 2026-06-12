"""Evaluate MaskWAM on the RoboTwin 2.0 benchmark.

Usage:
    python scripts/eval_robotwin.py --config configs/eval/robotwin.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MaskWAM on RoboTwin 2.0.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build model + Evaluator over RoboTwin tasks and report success rates.
    raise NotImplementedError("RoboTwin 2.0 evaluation is not released yet.")


if __name__ == "__main__":
    main()
