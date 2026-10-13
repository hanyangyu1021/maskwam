import numpy as np
import pytest

from platforms.real.infer.policy import denormalize_action_chunk
from platforms.real.infer.preprocessing import (
    concat_t_mask,
    concat_t_shape,
    consistent_front_crop,
    pad_last_dim,
)


def test_consistent_crop_keeps_image_and_mask_aligned():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.float32)
    cropped_image, cropped_mask = consistent_front_crop(image, mask)
    assert cropped_image.shape == (320, 432, 3)
    assert cropped_mask.shape == (320, 432)


def test_t_shape_layout():
    pytest.importorskip("cv2")
    front = np.zeros((320, 432, 3), dtype=np.uint8)
    wrist = np.zeros((480, 640, 3), dtype=np.uint8)
    combined = concat_t_shape(front, wrist, wrist)
    assert combined.shape == (480, 432, 3)
    mask = concat_t_mask(np.zeros((320, 432), dtype=np.float32))
    assert mask.shape == (480, 432)


def test_padding_rejects_truncation():
    assert pad_last_dim(np.ones((1, 16), dtype=np.float32), 32).shape == (1, 32)
    with pytest.raises(ValueError, match="超过"):
        pad_last_dim(np.ones((1, 33), dtype=np.float32), 32)


def test_first_action_chunk_shape_and_denormalization():
    normalized = np.ones((50, 32), dtype=np.float32)
    mean = np.arange(16, dtype=np.float32)
    std = np.full(16, 2.0, dtype=np.float32)
    actions = denormalize_action_chunk(normalized, mean, std, action_steps=48)
    assert actions.shape == (48, 16)
    np.testing.assert_allclose(actions[0], mean + 2)
