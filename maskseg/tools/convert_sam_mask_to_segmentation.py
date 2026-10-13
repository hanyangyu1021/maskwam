#!/usr/bin/env python3
"""Convert SAM mask NPZ files to segmentation_primary.npz format.

This script reads SAM-generated mask files from MaskWAM-Seg and converts
them into the ``segmentation_primary.npz`` format expected by the RoboInter
dataset loader, then places them into the corresponding
episode folders.

Format mapping:
  SAM mask (sam_mask/*.npz):
    - key "masks": shape (N_obj, T, 1, H, W), dtype bool

  segmentation_primary.npz:
    - key "instance_masks": shape (N_obj, T, H, W), dtype bool
    - key "union_mask":     shape (T, H, W),        dtype bool
    - key "original_shape": int64 array [N_obj, T_original, 1, H, W]

Matching logic:
  SAM mask filename: episode_XXXXX__<task>_cam_high.npz
  Episode folder:    episode_XXXXX__<task>/
  The mask filename without extension matches the video filename inside
  the episode folder (e.g. episode_00000__grasp-the-bottle_cam_high).

Typical usage (run from the MaskWAM-Seg root):

    python tools/convert_sam_mask_to_segmentation.py

    # Or specify paths explicitly:
    python tools/convert_sam_mask_to_segmentation.py \\
        --sam-mask-dir asserts/demo_data/human_anno_sam/0/sam_mask \\
        --episode-dir xtrainer_test/bowls
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SAM mask NPZ to segmentation_primary.npz format."
    )
    parser.add_argument(
        "--sam-mask-dir",
        type=Path,
        default="asserts/demo_data/human_anno_sam/0/sam_mask",
        help=(
            "Directory containing SAM mask .npz files. "
            "Default: <project>/MaskWAM-Seg/asserts/demo_data/human_anno_sam/0/sam_mask"
        ),
    )
    parser.add_argument(
        "--episode-dir",
        type=Path,
        default="xtrainer_test/bowls",
        help=(
            "Directory containing converted episode folders. "
            "Default: <project>/MaskWAM-Seg/xtrainer_test/bowls"
        ),
    )
    parser.add_argument(
        "--time",
        type=int,
        default=0,
        help="Annotation round (used to locate sam_mask dir when --sam-mask-dir is not set). Default: 0",
    )
    parser.add_argument(
        "--camera-suffix",
        type=str,
        default="_cam_high",
        help="Camera suffix in mask filenames. Default: _cam_high",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print what would be done without writing files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip episodes that already have segmentation_primary.npz.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def convert_sam_mask_to_segmentation_primary(
    sam_mask_path: Path,
    output_path: Path,
) -> dict:
    """Convert a single SAM mask NPZ to segmentation_primary.npz format.

    Parameters
    ----------
    sam_mask_path : Path
        Input SAM mask file (key "masks", shape (N_obj, T, 1, H, W)).
    output_path : Path
        Output path for segmentation_primary.npz.

    Returns
    -------
    dict with conversion metadata.
    """
    with np.load(sam_mask_path) as data:
        masks = data["masks"]  # (N_obj, T, 1, H, W) bool

    original_shape = np.array(masks.shape, dtype=np.int64)  # [N_obj, T, 1, H, W]

    # Squeeze the singleton dimension (axis=2) to get (N_obj, T, H, W)
    if masks.ndim == 5 and masks.shape[2] == 1:
        instance_masks = np.squeeze(masks, axis=2)
    elif masks.ndim == 4:
        # Already (N_obj, T, H, W)
        instance_masks = masks
    else:
        raise ValueError(
            f"Unexpected mask shape {masks.shape} in {sam_mask_path}. "
            f"Expected (N_obj, T, 1, H, W) or (N_obj, T, H, W)."
        )

    instance_masks = instance_masks.astype(bool, copy=False)
    union_mask = instance_masks.any(axis=0)  # (T, H, W)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        instance_masks=instance_masks,
        union_mask=union_mask,
        original_shape=original_shape,
    )

    return {
        "input_shape": masks.shape,
        "instance_masks_shape": instance_masks.shape,
        "union_mask_shape": union_mask.shape,
        "original_shape": original_shape.tolist(),
    }


# Known camera suffixes, ordered longest-first so greedy match works.
_KNOWN_CAM_SUFFIXES = [
    "_cam_left_wrist",
    "_cam_right_wrist",
    "_cam_high",
]


def detect_camera_suffix(mask_stem: str) -> Optional[str]:
    """Auto-detect camera suffix from mask filename stem.

    Returns the matched suffix string, or *None* if no known suffix is found.
    """
    for suffix in _KNOWN_CAM_SUFFIXES:
        if mask_stem.endswith(suffix):
            return suffix
    return None


def mask_filename_to_episode_dir_name(mask_stem: str, camera_suffix: str) -> Optional[str]:
    """Derive episode folder name from mask filename stem.

    Example:
        mask_stem = "episode_00000__grasp-the-bottle_cam_high"
        camera_suffix = "_cam_high"
        -> "episode_00000__grasp-the-bottle"
    """
    if mask_stem.endswith(camera_suffix):
        return mask_stem[: -len(camera_suffix)]
    return None


def segmentation_output_name(camera_suffix: str) -> str:
    """Return the output .npz filename based on camera suffix.

    _cam_high        -> segmentation_primary.npz
    _cam_left_wrist  -> segmentation_primary_left_wrist.npz
    _cam_right_wrist -> segmentation_primary_right_wrist.npz
    """
    if camera_suffix == "_cam_high":
        return "segmentation_primary.npz"
    # Strip leading "_cam_" and build name
    cam_tag = camera_suffix.lstrip("_").replace("cam_", "", 1)  # e.g. "left_wrist"
    return f"segmentation_primary_{cam_tag}.npz"


def canonical_video_name(camera_suffix: str) -> str:
    """Return the canonical short video filename for a camera.

    _cam_high        -> cam_high.mp4
    _cam_left_wrist  -> cam_left_wrist.mp4
    """
    return camera_suffix.lstrip("_") + ".mp4"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve default paths relative to the MaskWAM-Seg root.
    project_root = Path(__file__).resolve().parents[1]

    if args.sam_mask_dir is not None:
        sam_mask_dir = args.sam_mask_dir.resolve()
    else:
        sam_mask_dir = (
            project_root
            / "asserts"
            / "demo_data"
            / "human_anno_sam"
            / str(args.time)
            / "sam_mask"
        )

    if args.episode_dir is not None:
        episode_dir = args.episode_dir.resolve()
    else:
        episode_dir = project_root / "xtrainer_test" / "bowls"

    if not sam_mask_dir.exists():
        raise FileNotFoundError(f"SAM mask directory not found: {sam_mask_dir}")
    if not episode_dir.exists():
        raise FileNotFoundError(f"Episode directory not found: {episode_dir}")

    mask_files = sorted(sam_mask_dir.glob("*.npz"))
    if not mask_files:
        print(f"[warn] No .npz files found in {sam_mask_dir}")
        return

    # Build a lookup: episode_dir_name -> episode_dir_path
    episode_dirs = {d.name: d for d in episode_dir.iterdir() if d.is_dir()}

    print(
        f"[setup] sam_mask_dir={sam_mask_dir}\n"
        f"        episode_dir={episode_dir}\n"
        f"        mask_files={len(mask_files)}\n"
        f"        episode_dirs={len(episode_dirs)}\n"
        f"        camera_suffix={args.camera_suffix}\n"
        f"        dry_run={args.dry_run}\n"
        f"        skip_existing={args.skip_existing}",
        flush=True,
    )

    converted = 0
    skipped = 0
    not_matched = 0

    for mask_path in mask_files:
        mask_stem = mask_path.stem  # e.g. "episode_00000__grasp-the-bottle_cam_high"

        # Auto-detect camera suffix; fall back to CLI arg if detection fails.
        cam_suffix = detect_camera_suffix(mask_stem) or args.camera_suffix
        ep_dir_name = mask_filename_to_episode_dir_name(mask_stem, cam_suffix)

        if ep_dir_name is None:
            print(
                f"  [skip] Cannot derive episode name from '{mask_stem}' "
                f"(camera_suffix='{cam_suffix}')",
                flush=True,
            )
            not_matched += 1
            continue

        if ep_dir_name not in episode_dirs:
            print(f"  [skip] No matching episode folder for '{ep_dir_name}'", flush=True)
            not_matched += 1
            continue

        target_dir = episode_dirs[ep_dir_name]
        output_path = target_dir / segmentation_output_name(cam_suffix)

        if args.skip_existing and output_path.exists():
            print(f"  [skip] Already exists: {output_path}", flush=True)
            skipped += 1
            continue

        # Rename video: episode_XXXXX__task_cam_xxx.mp4 -> cam_xxx.mp4
        # so that the RoboInter dataset loader can find it.
        old_video = target_dir / f"{mask_stem}.mp4"
        new_video = target_dir / canonical_video_name(cam_suffix)
        rename_video = old_video.exists() and not new_video.exists()

        if args.dry_run:
            print(f"  [dry-run] {mask_path.name} -> {output_path}", flush=True)
            if rename_video:
                print(f"  [dry-run] rename {old_video.name} -> {new_video.name}", flush=True)
            converted += 1
            continue

        try:
            meta = convert_sam_mask_to_segmentation_primary(mask_path, output_path)
            print(
                f"  [ok] {mask_path.name} -> {output_path.name} "
                f"(instance_masks={meta['instance_masks_shape']}, "
                f"union_mask={meta['union_mask_shape']})",
                flush=True,
            )
            if rename_video:
                old_video.rename(new_video)
                print(f"  [ok] renamed {old_video.name} -> {new_video.name}", flush=True)
            elif not new_video.exists():
                print(
                    f"  [warn] video not found: expected {old_video.name} in {target_dir}",
                    flush=True,
                )
            converted += 1
        except Exception as exc:
            print(f"  [error] {mask_path.name}: {exc}", flush=True)

    print(
        f"\n[done] converted={converted} skipped={skipped} not_matched={not_matched}",
        flush=True,
    )


if __name__ == "__main__":
    main()
