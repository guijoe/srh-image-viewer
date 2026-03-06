"""SRH ImageViewer — Student module template (NumPy-only image processing).

How to use
----------
1) Create a folder:   src/modules/<your_folder_name>/
2) Put THIS file inside that folder and rename it to:
      <your_folder_name>_module.py
   Example: src/modules/ahmet_yilmaz/ahmet_yilmaz_module.py
3) Inside the file, replace <YOUR NAME> / <YOUR MODULE NAME>.
4) Run:
      cd src
      python main_app.py

Notes
-----
- The UI is built with PySide6 (matches the sample module style).
- Transformations below are implemented with NumPy only.
"""

from __future__ import annotations

import numpy as np
import imageio.v2 as imageio

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QStackedWidget,
    QPushButton,
    QDoubleSpinBox,
)

from modules.i_image_module import IImageModule


class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""

    def get_params(self) -> dict:
        raise NotImplementedError


class ContrastStretchParamsWidget(BaseParamsWidget):
    """UI for contrast stretching output range [a, b]."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Output Min (a):"))
        self.a = QDoubleSpinBox()
        self.a.setMinimum(0.0)
        self.a.setMaximum(255.0)
        self.a.setValue(0.0)
        layout.addWidget(self.a)

        layout.addWidget(QLabel("Output Max (b):"))
        self.b = QDoubleSpinBox()
        self.b.setMinimum(0.0)
        self.b.setMaximum(255.0)
        self.b.setValue(255.0)
        layout.addWidget(self.b)

        layout.addStretch(1)

    def get_params(self) -> dict:
        return {"a": float(self.a.value()), "b": float(self.b.value())}


class GammaParamsWidget(BaseParamsWidget):
    """UI for gamma correction."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma (γ):"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setMinimum(0.05)
        self.gamma.setMaximum(5.0)
        self.gamma.setSingleStep(0.05)
        self.gamma.setValue(1.0)
        layout.addWidget(self.gamma)
        layout.addStretch(1)

    def get_params(self) -> dict:
        return {"gamma": float(self.gamma.value())}


class UnsharpParamsWidget(BaseParamsWidget):
    """UI for unsharp masking."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Blur sigma (σ):"))
        self.sigma = QDoubleSpinBox()
        self.sigma.setMinimum(0.3)
        self.sigma.setMaximum(5.0)
        self.sigma.setSingleStep(0.1)
        self.sigma.setValue(1.0)
        layout.addWidget(self.sigma)

        layout.addWidget(QLabel("Amount:"))
        self.amount = QDoubleSpinBox()
        self.amount.setMinimum(0.0)
        self.amount.setMaximum(3.0)
        self.amount.setSingleStep(0.1)
        self.amount.setValue(1.0)
        layout.addWidget(self.amount)

        layout.addStretch(1)

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma.value()), "amount": float(self.amount.value())}



class StudentControlsWidget(QWidget):
    """Left-side controls: operation selector + parameter panel + Apply button."""

    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets: dict[str, BaseParamsWidget] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("\n### Control Panel\n"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Contrast Stretching": ContrastStretchParamsWidget,
            "Gamma Correction": GammaParamsWidget,
            "Unsharp Mask": UnsharpParamsWidget,
        }

        for op_name, widget_cls in operations.items():
            w = widget_cls()
            self.param_widgets[op_name] = w
            self.params_stack.addWidget(w)
            self.operation_selector.addItem(op_name)

        self.apply_btn = QPushButton("Apply Processing")
        layout.addWidget(self.apply_btn)

        self.apply_btn.clicked.connect(self._on_apply)
        self.operation_selector.currentTextChanged.connect(self._on_op_changed)

        layout.addStretch(1)

    def _on_op_changed(self, op_name: str):
        self.params_stack.setCurrentWidget(self.param_widgets[op_name])

    def _on_apply(self):
        op_name = self.operation_selector.currentText()
        params = self.param_widgets[op_name].get_params()
        params["operation"] = op_name
        self.process_requested.emit(params)


def _dtype_range(dtype: np.dtype) -> tuple[float, float]:
    """Return (min, max) for an image dtype."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return 0.0, 1.0


def _as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False)


def contrast_stretch(img: np.ndarray, a: float, b: float) -> np.ndarray:
    """Linear rescaling per-channel to map current min/max -> [a, b]."""
    out = _as_float(img)

    if out.ndim == 2:
        xmin, xmax = float(out.min()), float(out.max())
        if xmax == xmin:
            return img
        y = (out - xmin) * ((b - a) / (xmax - xmin)) + a
        y = np.clip(y, a, b)
        return y.astype(img.dtype)

    if out.ndim == 3:
        xmin = out.min(axis=(0, 1), keepdims=True)
        xmax = out.max(axis=(0, 1), keepdims=True)
        denom = (xmax - xmin)
        denom = np.where(denom == 0, 1.0, denom)
        y = (out - xmin) * ((b - a) / denom) + a
        y = np.clip(y, a, b)
        return y.astype(img.dtype)

    return img


def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
    """Power-law transform per-channel."""
    x = _as_float(img)
    lo, hi = _dtype_range(img.dtype)

    x_hat = (x - lo) / max(hi - lo, 1e-8)
    x_hat = np.clip(x_hat, 0.0, 1.0)
    y_hat = np.power(x_hat, gamma)
    y = y_hat * (hi - lo) + lo
    y = np.clip(y, lo, hi)
    return y.astype(img.dtype)


def _gaussian_kernel1d(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(ax * ax) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _convolve1d_along_axis(x: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1D convolution with edge padding (NumPy-only).

    Uses np.convolve along the chosen axis. This is not the fastest approach,
    but it is simple and acceptable for a student capstone.
    """
    radius = (len(k) - 1) // 2
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (radius, radius)
    xp = np.pad(x, pad_width, mode="edge")

    def _conv(v: np.ndarray) -> np.ndarray:
        return np.convolve(v, k, mode="valid")

    return np.apply_along_axis(_conv, axis, xp)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = int(np.ceil(3.0 * sigma))
    radius = max(radius, 1)
    k = _gaussian_kernel1d(sigma, radius)

    x = _as_float(img)
    if x.ndim == 2:
        tmp = _convolve1d_along_axis(x, k, axis=1)
        out = _convolve1d_along_axis(tmp, k, axis=0)
        return out.astype(img.dtype)

    if x.ndim == 3:
        channels = []
        for c in range(x.shape[2]):
            tmp = _convolve1d_along_axis(x[:, :, c], k, axis=1)
            out = _convolve1d_along_axis(tmp, k, axis=0)
            channels.append(out)
        out3 = np.stack(channels, axis=2)
        return out3.astype(img.dtype)

    return img


def unsharp_mask(img: np.ndarray, sigma: float, amount: float) -> np.ndarray:
    lo, hi = _dtype_range(img.dtype)
    x = _as_float(img)
    blur = _as_float(gaussian_blur(img, sigma))
    detail = x - blur
    y = x + amount * detail
    y = np.clip(y, lo, hi)
    return y.astype(img.dtype)


class StudentImageModule(IImageModule):
    """Replace the name strings below with your own."""

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "<YOUR MODULE NAME>"  

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = StudentControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            img = imageio.imread(file_path)
            metadata = {"name": file_path.split("/")[-1]}
            return True, img, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict | None) -> np.ndarray:
        if not params:
            return image_data
        op = params.get("operation")
        if op == "Contrast Stretching":
            a = float(params.get("a", 0.0))
            b = float(params.get("b", 255.0))
            return contrast_stretch(image_data, a, b)
        if op == "Gamma Correction":
            gamma = float(params.get("gamma", 1.0))
            return gamma_correction(image_data, gamma)
        if op == "Unsharp Mask":
            sigma = float(params.get("sigma", 1.0))
            amount = float(params.get("amount", 1.0))
            return unsharp_mask(image_data, sigma, amount)
        return image_data
