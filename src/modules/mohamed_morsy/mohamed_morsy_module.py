# src/modules/mohamed_morsy/mohamed_morsy_module.py

import numpy as np
import imageio.v3 as iio

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import Signal

from modules.i_image_module import IImageModule


# =========================
# NumPy-only helpers
# =========================

def _ensure_hwc(img: np.ndarray) -> np.ndarray:
    """Ensure image is HxWxC."""
    if img.ndim == 2:
        return img[:, :, None]
    return img

def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False)

def _clip_cast(img_f: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Clip to dtype range and cast back (important after sharpening)."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        img_f = np.clip(img_f, info.min, info.max)
    return img_f.astype(dtype)

def _reflect_pad(img_hwc: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(img_hwc, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")

def convolve2d_hwc(img_hwc: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Convolution per channel, reflect padding.
    img_hwc: HxWxC
    kernel: kxk odd
    """
    k = kernel.shape[0]
    assert kernel.shape == (k, k) and k % 2 == 1, "Kernel must be odd square."
    pad = k // 2

    img_f = _to_float(img_hwc)
    padded = _reflect_pad(img_f, pad)

    H, W, C = img_hwc.shape
    out = np.zeros((H, W, C), dtype=np.float32)

    # Loop over kernel only (clean + fast enough for small kernels)
    for i in range(k):
        for j in range(k):
            out += kernel[i, j] * padded[i:i+H, j:j+W, :]

    return out

def gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    sigma = max(0.2, float(sigma))
    assert ksize % 2 == 1 and ksize >= 3
    ax = np.arange(-(ksize // 2), ksize // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma * sigma))
    kernel /= np.sum(kernel)
    return kernel

def to_grayscale(img_hwc: np.ndarray) -> np.ndarray:
    """Return HxWx1 grayscale (luminance)."""
    if img_hwc.shape[2] == 1:
        return img_hwc
    rgb = _to_float(img_hwc[:, :, :3])
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray[:, :, None]

def sobel_magnitude(img_hwc: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude, returns HxWx1 float."""
    gray = to_grayscale(img_hwc)

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    gx = convolve2d_hwc(gray, kx)[:, :, 0]
    gy = convolve2d_hwc(gray, ky)[:, :, 0]
    mag = np.sqrt(gx * gx + gy * gy)

    return mag[:, :, None]

def gamma_correction(img_hwc: np.ndarray, gamma: float) -> np.ndarray:
    """I_out = I_in^gamma (implemented safely for uint8 images)."""
    gamma = float(gamma)
    img_f = _to_float(img_hwc)

    # assume uint8 most of the time
    if np.issubdtype(img_hwc.dtype, np.integer):
        maxv = float(np.iinfo(img_hwc.dtype).max)
    else:
        maxv = float(np.max(img_f)) if np.max(img_f) > 0 else 1.0

    x = np.clip(img_f / maxv, 0.0, 1.0)
    y = np.power(x, gamma) * maxv
    return y

def unsharp_mask(img_hwc: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    """I_sharp = I + amount*(I - blur(I))."""
    sigma = max(0.2, float(sigma))
    amount = max(0.0, float(amount))
    k = gaussian_kernel(ksize=7, sigma=sigma)
    blurred = convolve2d_hwc(img_hwc, k)
    img_f = _to_float(img_hwc)
    sharp = img_f + amount * (img_f - blurred)
    return sharp


# =========================
# Parameter widgets (Sample-style)
# =========================

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class GaussianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sigma (blur strength):"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.2, 20.0)
        self.sigma.setValue(1.2)
        self.sigma.setSingleStep(0.1)
        layout.addWidget(self.sigma)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma.value())}

class GammaParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma ( < 1 brighter, > 1 darker ):"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(0.05, 5.0)
        # Default tuned for your portrait (a bit dark)
        self.gamma.setValue(0.75)
        self.gamma.setSingleStep(0.05)
        layout.addWidget(self.gamma)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"gamma": float(self.gamma.value())}

class SharpenParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sigma (blur used for unsharp mask):"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.2, 10.0)
        self.sigma.setValue(1.0)
        self.sigma.setSingleStep(0.1)
        layout.addWidget(self.sigma)

        layout.addWidget(QLabel("Amount (sharpen strength):"))
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0.0, 5.0)
        # Default tuned for your portrait (nice detail boost)
        self.amount.setValue(1.2)
        self.amount.setSingleStep(0.1)
        layout.addWidget(self.amount)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma.value()), "amount": float(self.amount.value())}


# =========================
# Control widget (Sample-style)
# =========================

class MohamedMorsyControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Imaging Technologies - Mohamed Morsy</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Gaussian Blur (NumPy)": GaussianParamsWidget,
            "Sobel Edge Detect (NumPy)": NoParamsWidget,
            "Gamma Correction (NumPy)": GammaParamsWidget,
            "Sharpen / Unsharp Mask (NumPy)": SharpenParamsWidget,
        }

        for name, widget_class in operations.items():
            w = widget_class()
            self.params_stack.addWidget(w)
            self.param_widgets[name] = w
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)
        layout.addStretch()

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

        # set initial params widget
        self._on_operation_changed(self.operation_selector.currentText())

    def _on_apply_clicked(self):
        op = self.operation_selector.currentText()
        params = self.param_widgets[op].get_params()
        params["operation"] = op
        self.process_requested.emit(params)

    def _on_operation_changed(self, op: str):
        self.params_stack.setCurrentWidget(self.param_widgets[op])


# =========================
# Module class (Sample-style)
# =========================

class MohamedMorsyImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Imaging Technologies - Mohamed Morsy"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = MohamedMorsyControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            img = iio.imread(file_path)
            img = _ensure_hwc(img)
            metadata = {"name": file_path.split("/")[-1]}
            return True, img, metadata, None
        except Exception as e:
            print(f"[MohamedMorsy] Error loading image: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        img = _ensure_hwc(image_data)
        dtype = image_data.dtype
        op = params.get("operation", "")

        if op == "Gaussian Blur (NumPy)":
            sigma = float(params.get("sigma", 1.2))
            k = gaussian_kernel(ksize=7, sigma=sigma)
            out = convolve2d_hwc(img, k)
            return _clip_cast(out, dtype)

        if op == "Sobel Edge Detect (NumPy)":
            mag = sobel_magnitude(img)
            # Normalize to dtype range for display
            if np.issubdtype(dtype, np.integer):
                maxv = float(np.iinfo(dtype).max)
                mmax = float(np.max(mag)) if np.max(mag) > 0 else 1.0
                out = (mag / mmax) * maxv
                return _clip_cast(out, dtype)
            return mag.astype(dtype)

        if op == "Gamma Correction (NumPy)":
            gamma = float(params.get("gamma", 0.75))
            out = gamma_correction(img, gamma=gamma)
            return _clip_cast(out, dtype)

        if op == "Sharpen / Unsharp Mask (NumPy)":
            sigma = float(params.get("sigma", 1.0))
            amount = float(params.get("amount", 1.2))
            out = unsharp_mask(img, sigma=sigma, amount=amount)
            return _clip_cast(out, dtype)

        return image_data