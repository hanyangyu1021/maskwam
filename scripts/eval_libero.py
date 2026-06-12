"""Evaluate MaskWAM on the LIBERO benchmark.

Usage:
    python scripts/eval_libero.py --config configs/eval/libero.yaml
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MaskWAM on LIBERO.")
    parser.add_argument("--config", required=True, help="Path to an eval YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: build model + Evaluator over LIBERO suites and report success rates.
    raise NotImplementedError("LIBERO evaluation is not released yet.")


if __name__ == "__main__":
    main()
