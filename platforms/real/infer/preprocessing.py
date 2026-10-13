"""XTrainer three-camera preprocessing shared by playground and tests."""

from __future__ import annotations

import numpy as np


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"图像必须是 HWC 或 CHW 三维数组，实际 {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"图像最后一维必须为 3，实际 {array.shape}")
    if array.dtype != np.uint8:
        scale = 255.0 if float(array.max(initial=0)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def consistent_front_crop(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Apply the crop used by the released XTrainer dataset."""

    height, width = image.shape[:2]
    y1, y2 = min(95, height), max(min(height - 65, height), min(95, height))
    x1, x2 = min(104, width), max(min(width - 104, width), min(104, width))
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = None if mask is None else np.asarray(mask)[y1:y2, x1:x2]
    if cropped_image.size == 0:
        raise ValueError(f"consistent crop 后图像为空，原始尺寸为 {(height, width)}")
    return cropped_image, cropped_mask


def pink_bowls_crop(
    image: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    height, width = image.shape[:2]
    slices = (slice(height * 2 // 5, height * 9 // 10), slice(width // 8, width * 3 // 5))
    return image[slices], None if mask is None else np.asarray(mask)[slices]


def concat_t_shape(
    front: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
) -> np.ndarray:
    import cv2

    front = ensure_rgb(front)
    left_wrist = ensure_rgb(left_wrist)
    right_wrist = ensure_rgb(right_wrist)[::-1, ::-1].copy()
    height, width = front.shape[:2]
    target = (width // 2, height // 2)
    left = cv2.resize(left_wrist, target, interpolation=cv2.INTER_LINEAR)
    right = cv2.resize(right_wrist, target, interpolation=cv2.INTER_LINEAR)
    return np.vstack((front, np.hstack((left, right))))


def concat_t_mask(front_mask: np.ndarray) -> np.ndarray:
    """Build a T mask with black wrist regions."""

    mask = np.asarray(front_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = mask.max(axis=-1)
    height, width = mask.shape
    wrists = np.zeros((height // 2, width), dtype=np.float32)
    return np.vstack((mask, wrists))


def pad_last_dim(array: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(array)
    if source.shape[-1] > size:
        raise ValueError(f"输入维度 {source.shape[-1]} 超过模型维度 {size}")
    result = np.zeros((*source.shape[:-1], size), dtype=source.dtype)
    result[..., : source.shape[-1]] = source
    return result


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    point: tuple[int, int] | None = None,
) -> np.ndarray:
    import cv2

    result = ensure_rgb(image).astype(np.float32)
    binary = np.asarray(mask) > 0.5
    color = np.zeros_like(result)
    color[..., 1] = 255
    result[binary] = result[binary] * 0.55 + color[binary] * 0.45
    result = np.clip(result, 0, 255).astype(np.uint8)
    if point is not None:
        cv2.drawMarker(result, point, (255, 40, 40), cv2.MARKER_CROSS, 20, 2)
    return result
