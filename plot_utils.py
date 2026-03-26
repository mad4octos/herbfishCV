import numpy as np
import cv2

GOLDEN_RATIO_CONJUGATE = 0.618033988749895  # 1/φ per Ankerl
DEFAULT_S, DEFAULT_V = 0.5, 0.95  # suggested by the article


def draw_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    class_id: int,
    alpha: float = 0.15,
    color=None,
    binary_mask=False,
):
    """
    Draw a semi-transparent colored overlay on regions of an image where the mask equals a given class ID.
    """
    if color is None:
        color = color_from_index_bgr(class_id)
    overlay = np.copy(image)

    if binary_mask:
        fg = mask != 0
    else:
        fg = mask == class_id

    overlay[fg] = color
    weighted_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return weighted_image


def hsv_to_rgb_ankerl(h: float, s: float, v: float) -> tuple[int, int, int]:
    """
    HSV in [0,1) -> RGB in 0..255.

    Modified from:
    https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
    """
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q

    return int(r * 256) % 256, int(g * 256) % 256, int(b * 256) % 256


def color_from_index(
    idx: int, h0: float = 0.0, s: float = DEFAULT_S, v: float = DEFAULT_V
) -> tuple[int, int, int]:
    """
    Deterministic color for any non-negative integer.
    Consecutive indices are well-separated by stepping hue with 1/φ.
    """
    # distribute hues by adding the golden-ratio conjugate and wrapping
    h = (h0 + idx * GOLDEN_RATIO_CONJUGATE) % 1.0
    return hsv_to_rgb_ankerl(h, s, v)


def color_from_index_bgr(idx: int, **kwargs) -> tuple[int, int, int]:
    """
    Same as color_from_index but returns BGR (handy for OpenCV).
    """
    r, g, b = color_from_index(idx, **kwargs)
    return (b, g, r)
