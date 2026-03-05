import numpy as np

def red_highlight(image_data: np.ndarray) -> np.ndarray:
    """
    Keep red pixels colored while converting everything else to grayscale.
    Red detection uses simple RGB thresholding.
    """
    img = image_data.copy()

    # If not RGB, just return as-is
    if img.ndim != 3 or img.shape[2] < 3:
        return image_data

    R = img[:, :, 0].astype(np.int16)
    G = img[:, :, 1].astype(np.int16)
    B = img[:, :, 2].astype(np.int16)

    # red mask: red is strong + higher than green/blue
    red_mask = (R > 120) & (R > G + 40) & (R > B + 40)

    # grayscale background (luma)
    gray = (0.299 * R + 0.587 * G + 0.114 * B).astype(np.uint8)

    result = img.copy()
    gray_rgb = np.stack([gray, gray, gray], axis=-1)
    result[~red_mask] = gray_rgb[~red_mask]

    return result.astype(image_data.dtype)