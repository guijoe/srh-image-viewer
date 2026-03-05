import numpy as np
from PyQt6.QtWidgets import (
    QVBoxLayout, QLabel, QDoubleSpinBox, QSpinBox
)
from modules.i_image_module import IImageModule, BaseControlsWidget, BaseParamsWidget, NoParamsWidget


# ─────────────────────────────────────────────
#  PARAMETER WIDGETS
# ─────────────────────────────────────────────

class ContrastStretchingParamsWidget(BaseParamsWidget):
    """Widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'new_min': self.min_spinbox.value(),
            'new_max': self.max_spinbox.value()
        }


class BrightnessParamsWidget(BaseParamsWidget):
    """Widget for Brightness Adjustment parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness Offset (-255 to 255):"))
        self.offset_spinbox = QSpinBox()
        self.offset_spinbox.setMinimum(-255)
        self.offset_spinbox.setMaximum(255)
        self.offset_spinbox.setValue(50)
        layout.addWidget(self.offset_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'offset': self.offset_spinbox.value()}


class GaussianBlurParamsWidget(BaseParamsWidget):
    """Widget for Gaussian Blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Kernel Size (odd number, 3-15):"))
        self.kernel_spinbox = QSpinBox()
        self.kernel_spinbox.setMinimum(3)
        self.kernel_spinbox.setMaximum(15)
        self.kernel_spinbox.setSingleStep(2)
        self.kernel_spinbox.setValue(5)
        layout.addWidget(self.kernel_spinbox)

        layout.addWidget(QLabel("Sigma (standard deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setMaximum(10.0)
        self.sigma_spinbox.setSingleStep(0.1)
        self.sigma_spinbox.setValue(1.0)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        k = self.kernel_spinbox.value()
        if k % 2 == 0:
            k += 1
        return {'kernel_size': k, 'sigma': self.sigma_spinbox.value()}


class SharpenParamsWidget(BaseParamsWidget):
    """Widget for Sharpening parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sharpening Strength (0.5 - 5.0):"))
        self.strength_spinbox = QDoubleSpinBox()
        self.strength_spinbox.setMinimum(0.5)
        self.strength_spinbox.setMaximum(5.0)
        self.strength_spinbox.setSingleStep(0.5)
        self.strength_spinbox.setValue(1.5)
        layout.addWidget(self.strength_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'strength': self.strength_spinbox.value()}


# ─────────────────────────────────────────────
#  CONTROLS WIDGET
# ─────────────────────────────────────────────

class DaniKabbaniControlsWidget(BaseControlsWidget):
    def setup_ui(self):
        operations = {
            "Grayscale":            NoParamsWidget,
            "Negative":             NoParamsWidget,
            "Contrast Stretching":  ContrastStretchingParamsWidget,
            "Brightness Adjustment": BrightnessParamsWidget,
            "Gaussian Blur":        GaussianBlurParamsWidget,
            "Sharpen":              SharpenParamsWidget,
        }
        self.setup_operations(operations)


# ─────────────────────────────────────────────
#  IMAGE PROCESSING LOGIC
# ─────────────────────────────────────────────

def grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale using the luminosity formula.
    Weights: R=0.2989, G=0.5870, B=0.1140 (ITU-R BT.601)
    The result is returned as a 3-channel image so it remains compatible
    with the rest of the application.
    """
    if image.ndim == 2:
        return image  # Already grayscale
    weights = np.array([0.2989, 0.5870, 0.1140])
    gray = np.sum(image[..., :3] * weights, axis=2).astype(image.dtype)
    # Stack back to 3 channels
    return np.stack([gray, gray, gray], axis=2)


def negative(image: np.ndarray) -> np.ndarray:
    """
    Compute the negative of an image.
    Formula: output = 255 - input  (for 8-bit images)
    """
    max_val = np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 1.0
    return (max_val - image).astype(image.dtype)


def contrast_stretching(image: np.ndarray, new_min: float, new_max: float) -> np.ndarray:
    """
    Linear contrast stretching.
    Maps [current_min, current_max] → [new_min, new_max] using:
        output = (input - curr_min) * (new_max - new_min) / (curr_max - curr_min) + new_min
    """
    img_float = image.astype(float)
    curr_min = np.min(img_float)
    curr_max = np.max(img_float)

    if curr_max == curr_min:
        return image  # Flat image, nothing to stretch

    stretched = (img_float - curr_min) * ((new_max - new_min) / (curr_max - curr_min)) + new_min
    stretched = np.clip(stretched, new_min, new_max)
    return stretched.astype(image.dtype)


def brightness_adjustment(image: np.ndarray, offset: int) -> np.ndarray:
    """
    Add a constant offset to every pixel.
    Values are clipped to stay within [0, 255].
    Formula: output = clip(input + offset, 0, 255)
    """
    img_int = image.astype(np.int32) + offset
    img_clipped = np.clip(img_int, 0, 255)
    return img_clipped.astype(image.dtype)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Build a 2D Gaussian kernel of given size and sigma.
    Each element: G(x,y) = exp(-(x²+y²) / (2σ²))
    The kernel is then normalised so all weights sum to 1.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur(image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian blur via 2D convolution.
    Each output pixel is a weighted average of its neighbours,
    with weights following a Gaussian (bell-curve) distribution.
    Edges are handled with 'reflect' padding.
    """
    kernel = _gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2

    if image.ndim == 2:
        channels = [image]
    else:
        channels = [image[..., c] for c in range(image.shape[2])]

    blurred_channels = []
    for ch in channels:
        padded = np.pad(ch, pad, mode='reflect').astype(float)
        output = np.zeros_like(ch, dtype=float)
        for i in range(ch.shape[0]):
            for j in range(ch.shape[1]):
                region = padded[i:i + kernel_size, j:j + kernel_size]
                output[i, j] = np.sum(region * kernel)
        blurred_channels.append(output)

    if image.ndim == 2:
        result = blurred_channels[0]
    else:
        result = np.stack(blurred_channels, axis=2)

    return np.clip(result, 0, 255).astype(image.dtype)


def sharpen(image: np.ndarray, strength: float) -> np.ndarray:
    """
    Unsharp masking sharpening.
    Steps:
      1. Blur the image with a Gaussian kernel.
      2. Compute the 'mask' = original - blurred  (the edge detail).
      3. Add scaled mask back: output = original + strength * mask
    The result highlights edges and fine details.
    """
    blurred = gaussian_blur(image, kernel_size=5, sigma=1.0).astype(float)
    original = image.astype(float)
    mask = original - blurred
    sharpened = original + strength * mask
    return np.clip(sharpened, 0, 255).astype(image.dtype)


# ─────────────────────────────────────────────
#  MODULE CLASS
# ─────────────────────────────────────────────

class DaniKabbaniImageModule(IImageModule):

    def get_module_name(self) -> str:
        return "Dani Kabbani"

    def get_controls_widget(self):
        return DaniKabbaniControlsWidget()

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed = image_data.copy()
        operation = params.get('operation')

        if operation == "Grayscale":
            processed = grayscale(processed)

        elif operation == "Negative":
            processed = negative(processed)

        elif operation == "Contrast Stretching":
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)
            processed = contrast_stretching(processed, new_min, new_max)

        elif operation == "Brightness Adjustment":
            offset = params.get('offset', 50)
            processed = brightness_adjustment(processed, offset)

        elif operation == "Gaussian Blur":
            kernel_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 1.0)
            processed = gaussian_blur(processed, kernel_size, sigma)

        elif operation == "Sharpen":
            strength = params.get('strength', 1.5)
            processed = sharpen(processed, strength)

        return processed
