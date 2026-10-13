import numpy as np

from platforms.real.infer.preprocessing import consistent_front_crop, overlay_mask


def test_click_stays_on_original_image_before_crop():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.float32)
    point = (200, 180)
    mask[point[1] - 4 : point[1] + 5, point[0] - 4 : point[0] + 5] = 1.0

    overlay = overlay_mask(image, mask, point)
    assert overlay.shape == image.shape
    assert overlay[point[1], point[0], 0] > 200

    cropped_image, cropped_mask = consistent_front_crop(image, mask)
    assert cropped_image.shape[0] < image.shape[0]
    assert cropped_mask.shape == cropped_image.shape[:2]
