"""Minimal real-robot deployment example for MaskWAM.

This script shows the intended closed-loop control interface: read camera
observations, (optionally) supply a first-frame target mask, query the policy
for an action chunk, and stream actions to the robot controller.

Usage:
    python deploy/real_robot_demo.py --checkpoint checkpoints/maskwam_real/last.ckpt
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MaskWAM real-robot demo.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--instruction", default="", help="Language instruction.")
    parser.add_argument("--camera", default="0", help="Camera id / RTSP url.")
    parser.add_argument("--control-hz", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: connect to camera + robot SDK, build model, run the control loop.
    raise NotImplementedError("Real-robot deployment example is not released yet.")


if __name__ == "__main__":
    main()
