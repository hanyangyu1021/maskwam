"""Small SAM3 point-prompt adapter using the repository's single SAM3 copy."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def _load_sam3_builder():
    try:
        from sam3.model_builder import build_sam3_video_model

        return build_sam3_video_model
    except ModuleNotFoundError as error:
        if error.name != "sam3":
            raise

    project_root = Path(__file__).resolve().parents[3]
    sam3_root = Path(
        os.environ.get("MASKWAM_SAM3_ROOT", project_root / "third_party" / "sam3")
    ).expanduser().resolve()
    if not sam3_root.is_dir():
        raise FileNotFoundError(
            f"SAM3 source does not exist: {sam3_root}. Install the sam3 package "
            "or set MASKWAM_SAM3_ROOT to its repository root."
        )
    sys.path.insert(0, str(sam3_root))
    from sam3.model_builder import build_sam3_video_model

    return build_sam3_video_model


class SAM3PointSegmenter:
    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        confidence_threshold: float = 0.3,
    ) -> None:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"SAM3 checkpoint 不存在: {checkpoint}. "
                "请在 Playground 中设置正确的 sam3.pt 路径。"
            )
        build_sam3_video_model = _load_sam3_builder()

        self.device = device
        self.model = build_sam3_video_model(
            checkpoint_path=str(checkpoint),
            load_from_HF=False,
            device=device,
        )
        self.predictor = self.model.tracker
        self.predictor.backbone = self.model.detector.backbone

    def _autocast(self):
        if self.device.startswith("cuda"):
            import torch

            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def segment(self, image: np.ndarray, point: tuple[int, int]) -> np.ndarray:
        import cv2
        import torch

        rgb = np.asarray(image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        normalized_point = torch.tensor(
            [[point[0] / width, point[1] / height]],
            dtype=torch.float32,
        )
        labels = torch.tensor([1], dtype=torch.int32)

        with tempfile.NamedTemporaryFile(suffix=".mp4") as handle:
            writer = cv2.VideoWriter(
                handle.name,
                cv2.VideoWriter_fourcc(*"mp4v"),
                1,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建 SAM3 单帧临时视频")
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            writer.release()

            state = self.predictor.init_state(video_path=handle.name)
            self.predictor.clear_all_points_in_video(state)
            with torch.inference_mode(), self._autocast():
                _, object_ids, _, video_masks = self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=1,
                    points=normalized_point,
                    labels=labels,
                    clear_old_points=False,
                )

        if len(object_ids) == 0 or video_masks is None or len(video_masks) == 0:
            raise RuntimeError("SAM3 没有返回 mask，请重新点击目标物体内部")
        return (video_masks[0].squeeze().float().cpu().numpy() > 0).astype(np.float32)
