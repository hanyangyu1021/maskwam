"""Run MaskWAM inference on a single episode / observation sequence.

Usage:
    python scripts/inference.py --checkpoint checkpoints/maskwam_libero/last.ckpt \\
        --obs path/to/obs.npz --mask path/to/target_mask.png
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MaskWAM inference.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs", required=True, help="Observation file (npz / video).")
    parser.add_argument("--instruction", default="", help="Language instruction.")
    parser.add_argument("--mask", default=None, help="Optional first-frame target mask.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load checkpoint, build model, run model.predict_action loop.
    raise NotImplementedError("Inference entry point is not released yet.")


if __name__ == "__main__":
    main()
