from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


# ----------------- NumPy-only helpers -----------------
def _to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 and normalize to [0,1]."""
    img = img.astype(np.float32)
    if img.max() > 1.5:  # likely uint8-like (0..255)
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)

def _to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert [0,1] float image back to uint8 [0,255]."""
    return (np.clip(img01, 0.0, 1.0) * 255.0).astype(np.uint8)

def _apply_per_channel(img: np.ndarray, fn):
    """Apply fn to grayscale or RGB/RGBA (preserve alpha)."""
    if img.ndim == 2:
        return fn(img)
    if img.ndim == 3:
        c = img.shape[2]
        if c == 4:  # RGBA
            rgb = fn(img[:, :, :3])
            out = img.copy()
            out[:, :, :3] = rgb
            return out
        return fn(img)
    return img


# ----------------- 3 transformations -----------------
def brightness_transform(img: np.ndarray, delta: float) -> np.ndarray:
    """
    Brightness: I_out = clip(I_in + delta, 0, 255)
    Implemented in normalized domain.
    """
    img01 = _to_float01(img)
    out01 = img01 + (delta / 255.0)
    return _to_uint8(out01)

def gamma_transform(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Gamma correction: I_out = I_in ^ gamma (on normalized [0,1]).
    gamma < 1 -> brighter, gamma > 1 -> darker.
    """
    img01 = _to_float01(img)
    out01 = np.power(img01, gamma)
    return _to_uint8(out01)

def contrast_stretch_transform(img: np.ndarray) -> np.ndarray:
    """
    Contrast stretching (min-max normalization):
      I_out = (I - I_min) / (I_max - I_min) * 255
    Done per-channel (RGB) and preserves alpha (RGBA).
    """
    def fn(x):
        x_f = x.astype(np.float32)

        # If grayscale
        if x_f.ndim == 2:
            mn = np.min(x_f)
            mx = np.max(x_f)
            if mx - mn < 1e-9:
                return x.astype(np.uint8) if x.dtype != np.uint8 else x
            out = (x_f - mn) / (mx - mn)
            return _to_uint8(out)

        # RGB: per channel stretch
        chs = []
        for i in range(x_f.shape[2]):
            ch = x_f[:, :, i]
            mn = np.min(ch)
            mx = np.max(ch)
            if mx - mn < 1e-9:
                chs.append(ch.astype(np.uint8))
            else:
                out = (ch - mn) / (mx - mn)
                chs.append(_to_uint8(out))
        return np.stack(chs, axis=2)

    return _apply_per_channel(img, fn)


# ----------------- Parameter widgets -----------------
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

class BrightnessParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness delta (0..120):"))
        self.delta = QSpinBox()
        self.delta.setMinimum(0)
        self.delta.setMaximum(120)
        self.delta.setValue(40)
        layout.addWidget(self.delta)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"delta": int(self.delta.value())}

class GammaParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma (0.2..3.0):"))
        self.gamma = QDoubleSpinBox()
        self.gamma.setMinimum(0.2)
        self.gamma.setMaximum(3.0)
        self.gamma.setValue(1.2)
        self.gamma.setSingleStep(0.1)
        layout.addWidget(self.gamma)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"gamma": float(self.gamma.value())}


# ----------------- Controls UI -----------------
class SukhControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Sukh Controls</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.op = QComboBox()
        layout.addWidget(self.op)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        operations = {
            "Brightness": BrightnessParamsWidget,
            "Gamma": GammaParamsWidget,
            "Contrast Stretching": NoParamsWidget,
        }

        for name, widget_cls in operations.items():
            w = widget_cls()
            self.stack.addWidget(w)
            self.param_widgets[name] = w
            self.op.addItem(name)

        self.apply_btn = QPushButton("Apply Processing")
        layout.addWidget(self.apply_btn)

        self.apply_btn.clicked.connect(self._apply)
        self.op.currentTextChanged.connect(self._on_op_changed)

    def _on_op_changed(self, name: str):
        self.stack.setCurrentWidget(self.param_widgets[name])

    def _apply(self):
        name = self.op.currentText()
        params = self.param_widgets[name].get_params()
        params["operation"] = name
        self.process_requested.emit(params)


# ----------------- The actual module -----------------
class SukhImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls = None

    def get_name(self) -> str:
        return "Sukh Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls is None:
            self._controls = SukhControlsWidget(module_manager, parent)
            self._controls.process_requested.connect(self._handle_processing_request)
        return self._controls

    def _handle_processing_request(self, params: dict):
        if self._controls and self._controls.module_manager:
            self._controls.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            img = imageio.imread(file_path)
            metadata = {"name": file_path.split("/")[-1]}
            return True, img, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get("operation", "")

        if op == "Brightness":
            delta = params.get("delta", 40)
            return brightness_transform(image_data, float(delta))

        if op == "Gamma":
            gamma = params.get("gamma", 1.2)
            return gamma_transform(image_data, float(gamma))

        if op == "Contrast Stretching":
            return contrast_stretch_transform(image_data)

        return image_data