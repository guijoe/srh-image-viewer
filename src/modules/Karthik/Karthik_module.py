from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
import imageio
from numpy.lib.stride_tricks import sliding_window_view

from modules.i_image_module import IImageModule


# ---------------- Parameter Widgets ----------------
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """A placeholder widget for operations with no parameters."""
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
    """Gaussian blur parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sigma (0.5–5.0):"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setMinimum(0.5)
        self.sigma_spin.setMaximum(5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)
        layout.addWidget(self.sigma_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma_spin.value())}


class UnsharpParamsWidget(BaseParamsWidget):
    """Unsharp masking parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sigma for blur (0.5–5.0):"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setMinimum(0.5)
        self.sigma_spin.setMaximum(5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)
        layout.addWidget(self.sigma_spin)

        layout.addWidget(QLabel("Amount k (0.0–3.0):"))
        self.amount_spin = QDoubleSpinBox()
        self.amount_spin.setMinimum(0.0)
        self.amount_spin.setMaximum(3.0)
        self.amount_spin.setSingleStep(0.1)
        self.amount_spin.setValue(1.0)
        layout.addWidget(self.amount_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "sigma": float(self.sigma_spin.value()),
            "amount": float(self.amount_spin.value())
        }


class GammaParamsWidget(BaseParamsWidget):
    """Gamma correction parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma (0.1–5.0):"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setMinimum(0.1)
        self.gamma_spin.setMaximum(5.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        layout.addWidget(self.gamma_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"gamma": float(self.gamma_spin.value())}


# ---------------- Controls Widget ----------------
class karthikControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Control Panel</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # ✅ 5 operations (3 original + 2 new)
        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Unsharp Masking": UnsharpParamsWidget,
            "Gamma Correction": GammaParamsWidget,
            "Negative": NoParamsWidget,                # NEW
            "Contrast Stretching": NoParamsWidget,     # NEW
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

        self._on_operation_changed(self.operation_selector.currentText())

    def _on_apply_clicked(self):
        operation_name = self.operation_selector.currentText()
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params["operation"] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# ---------------- Module Implementation ----------------
class karthikImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "karthik Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = karthikControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        """Uses imageio for loading (allowed). Processing algorithms are NumPy-only."""
        try:
            image_data = imageio.imread(file_path)
            metadata = {"name": file_path.split("/")[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    # ---------------- NumPy-only helpers ----------------
    def _to_float01(self, img: np.ndarray) -> np.ndarray:
        """Convert image to float32 in [0,1] using best-effort normalization."""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        x = img.astype(np.float32)
        if float(np.max(x)) > 1.5:
            x = x / 255.0
        return np.clip(x, 0.0, 1.0)

    def _from_float01(self, x01: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """Convert float32 [0,1] back to original dtype."""
        x01 = np.clip(x01, 0.0, 1.0)
        if dtype == np.uint8:
            return (x01 * 255.0 + 0.5).astype(np.uint8)
        return x01.astype(dtype)

    def _apply_gray_or_rgb(self, img01: np.ndarray, fn_gray):
        """
        Apply a 2D gray function to:
          - grayscale (H,W)
          - RGB (H,W,3) per channel
          - RGBA (H,W,4) per channel (preserve alpha)
        Operates in float [0,1] domain.
        """
        # Special-case grayscale sometimes represented as (1,H,W)
        if img01.ndim == 3 and img01.shape[0] == 1 and (img01.shape[2] not in (3, 4)):
            out2d = fn_gray(img01[0])
            return out2d[np.newaxis, :, :]

        if img01.ndim == 2:
            return fn_gray(img01)

        if img01.ndim == 3 and img01.shape[2] in (3, 4):
            out = img01.copy()
            for c in range(3):
                out[:, :, c] = fn_gray(img01[:, :, c])
            # alpha remains unchanged if present
            return out

        return img01

    # ---- Separable Gaussian (fast) ----
    def _gaussian_kernel_1d(self, sigma: float) -> np.ndarray:
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        radius = size // 2
        x = np.arange(-radius, radius + 1, dtype=np.float32)
        k = np.exp(-(x * x) / (2.0 * sigma * sigma))
        k /= np.sum(k)
        return k.astype(np.float32)

    def _convolve1d_axis_fast(self, img2d: np.ndarray, kernel1d: np.ndarray, axis: int) -> np.ndarray:
        """
        Fast 1D convolution along axis 0 or 1 using sliding_window_view + tensordot.
        Reflect padding. img2d float32 2D.
        """
        k = int(kernel1d.shape[0])
        pad = k // 2

        if axis == 1:
            padded = np.pad(img2d, ((0, 0), (pad, pad)), mode="reflect")
            windows = sliding_window_view(padded, (k,), axis=1)  # (H, W, k)
            return np.tensordot(windows, kernel1d, axes=([2], [0])).astype(np.float32)

        if axis == 0:
            padded = np.pad(img2d, ((pad, pad), (0, 0)), mode="reflect")
            windows = sliding_window_view(padded, (k,), axis=0)  # (H, W, k)
            return np.tensordot(windows, kernel1d, axes=([2], [0])).astype(np.float32)

        raise ValueError("axis must be 0 or 1")

    def _gaussian_blur_gray(self, g01: np.ndarray, sigma: float) -> np.ndarray:
        """Gaussian blur for one grayscale channel in float [0,1]."""
        k1d = self._gaussian_kernel_1d(sigma)
        g = g01.astype(np.float32)
        tmp = self._convolve1d_axis_fast(g, k1d, axis=1)   # horizontal
        out = self._convolve1d_axis_fast(tmp, k1d, axis=0) # vertical
        return out

    # Main Process function
    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get("operation", "")
        dtype_in = image_data.dtype
        img01 = self._to_float01(image_data)

        if op == "Gaussian Blur":
            sigma = float(params.get("sigma", 1.0))
            if sigma <= 0:
                return image_data
            sigma = min(sigma, 5.0)
            out01 = self._apply_gray_or_rgb(img01, lambda g: self._gaussian_blur_gray(g, sigma))
            return self._from_float01(out01, dtype_in)

        if op == "Unsharp Masking":
            sigma = float(params.get("sigma", 1.0))
            amount = float(params.get("amount", 1.0))
            if sigma <= 0 or amount <= 0:
                return image_data
            sigma = min(sigma, 5.0)

            blurred = self._apply_gray_or_rgb(img01, lambda g: self._gaussian_blur_gray(g, sigma))
            sharpened = img01 + amount * (img01 - blurred)
            sharpened = np.clip(sharpened, 0.0, 1.0)
            return self._from_float01(sharpened, dtype_in)

        if op == "Gamma Correction":
            gamma = float(params.get("gamma", 1.0))
            if gamma <= 0:
                return image_data
            out01 = np.power(np.clip(img01, 0.0, 1.0), gamma).astype(np.float32)
            return self._from_float01(out01, dtype_in)

        if op == "Negative":
            out01 = self._apply_gray_or_rgb(img01, lambda g: 1.0 - np.clip(g, 0.0, 1.0))
            return self._from_float01(out01, dtype_in)

        if op == "Contrast Stretching":
            def stretch_gray(g):
                gmin = float(np.min(g))
                gmax = float(np.max(g))
                if gmax <= gmin:
                    return g
                return (g - gmin) / (gmax - gmin)

            out01 = self._apply_gray_or_rgb(img01, stretch_gray)
            return self._from_float01(out01, dtype_in)

        return image_data