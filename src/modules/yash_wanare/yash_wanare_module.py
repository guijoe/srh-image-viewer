# -*- coding: utf-8 -*-
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox, QSpinBox
)
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
from modules.i_image_module import IImageModule
from modules.sample.sample_module import BaseParamsWidget


# ── PARAM WIDGETS ─────────────────────────────────────────────────────────────

class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters needed. Just click Apply!"))
        layout.addStretch()
    def get_params(self) -> dict:
        return {}

class BrightnessParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Brightness Value (Min: -255 | Max: 255):"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(-255.0, 255.0)
        self.val_spin.setValue(30.0)
        layout.addWidget(self.val_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'value': self.val_spin.value()}

class ThresholdParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Threshold Level (Min: 0 | Max: 255):"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 255)
        self.val_spin.setValue(127)
        layout.addWidget(self.val_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'threshold': self.val_spin.value()}

class PowerLawParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma (Min: 0.01 | Max: 5.0):"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 5.0)
        self.gamma_spin.setValue(1.5)
        self.gamma_spin.setSingleStep(0.1)
        layout.addWidget(self.gamma_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'gamma': self.gamma_spin.value()}

class ContrastStretchParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spin = QDoubleSpinBox()
        self.min_spin.setRange(0.0, 255.0)
        self.min_spin.setValue(0.0)
        layout.addWidget(self.min_spin)
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0.0, 255.0)
        self.max_spin.setValue(255.0)
        layout.addWidget(self.max_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'new_min': self.min_spin.value(), 'new_max': self.max_spin.value()}

class SaltPepperParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Noise Density (0.0 - 0.5):"))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(0.0, 0.5)
        self.density_spin.setValue(0.05)
        self.density_spin.setSingleStep(0.01)
        layout.addWidget(self.density_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'density': self.density_spin.value()}

class RotationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Rotation Angle:"))
        self.angle_combo = QComboBox()
        self.angle_combo.addItems(["90", "180", "270"])
        layout.addWidget(self.angle_combo)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'angle': self.angle_combo.currentText()}


# ── CONTROLS WIDGET ───────────────────────────────────────────────────────────

class YashWanareControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.descriptions = {
            "1. Color Inversion (Negative)": "Inverts every pixel: output = 255 - input.",
            "2. Brightness Adjustment":      "Adds a constant value to all pixels to lighten or darken.",
            "3. Binary Thresholding":        "Converts image to black & white based on a threshold level.",
            "4. Power Law (Gamma)":          "Non-linearly maps intensities using output = c * input^gamma.",
            "5. Contrast Stretching":        "Linearly stretches pixel intensity range to [new_min, new_max].",
            "6. Histogram Equalization":     "Redistributes pixel intensities using the CDF for better contrast.",
            "7. Horizontal Flip":            "Mirrors the image left-to-right.",
            "8. Vertical Flip":              "Mirrors the image top-to-bottom.",
            "9. Salt & Pepper Noise":        "Randomly sets pixels to 0 (pepper) or 255 (salt).",
            "10. Image Rotation":            "Rotates the image 90, 180, or 270 degrees.",
        }
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        header = QLabel("Yash Wanare - Image Processing Module")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        layout.addWidget(header)

        layout.addWidget(QLabel("Select Operation:"))
        self.op_combo = QComboBox()

        operations = {
            "1. Color Inversion (Negative)": NoParamsWidget,
            "2. Brightness Adjustment":      BrightnessParamsWidget,
            "3. Binary Thresholding":        ThresholdParamsWidget,
            "4. Power Law (Gamma)":          PowerLawParamsWidget,
            "5. Contrast Stretching":        ContrastStretchParamsWidget,
            "6. Histogram Equalization":     NoParamsWidget,
            "7. Horizontal Flip":            NoParamsWidget,
            "8. Vertical Flip":              NoParamsWidget,
            "9. Salt & Pepper Noise":        SaltPepperParamsWidget,
            "10. Image Rotation":            RotationParamsWidget,
        }

        self.stacked_widget = QStackedWidget()
        for name, WidgetClass in operations.items():
            self.op_combo.addItem(name)
            w = WidgetClass()
            self.param_widgets[name] = w
            self.stacked_widget.addWidget(w)

        self.op_combo.currentTextChanged.connect(self._on_op_changed)
        layout.addWidget(self.op_combo)
        layout.addWidget(self.stacked_widget)

        self.desc_label = QLabel(self.descriptions.get(self.op_combo.currentText(), ""))
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet("color: grey; font-style: italic; font-size: 11px;")
        layout.addWidget(self.desc_label)

        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(self._on_apply)
        layout.addWidget(self.apply_btn)
        layout.addStretch()

    def _on_op_changed(self, name):
        if name in self.param_widgets:
            self.stacked_widget.setCurrentWidget(self.param_widgets[name])
        self.desc_label.setText(self.descriptions.get(name, ""))

    def _on_apply(self):
        op = self.op_combo.currentText()
        params = self.param_widgets[op].get_params()
        params['operation'] = op
        self.process_requested.emit(params)


# ── MAIN MODULE CLASS ─────────────────────────────────────────────────────────

class YashWanareImageModule(IImageModule):

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Yash Wanare Module"

    def get_supported_formats(self) -> list:
        return ["png", "jpg", "jpeg", "bmp", "tiff", "tif"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = YashWanareControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split("/")[-1].split("\\")[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _equalize_channel(self, channel):
        hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0].min()
        total = channel.size
        lut = np.round((cdf - cdf_min) / (total - cdf_min) * 255).astype(np.uint8)
        return lut[channel]

    # ── process ───────────────────────────────────────────────────────────────

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        img = image_data.copy()
        op = params.get('operation', '')

        if op == "1. Color Inversion (Negative)":
            img = 255 - img

        elif op == "2. Brightness Adjustment":
            val = params.get('value', 30)
            img = np.clip(img.astype(float) + val, 0, 255).astype(image_data.dtype)

        elif op == "3. Binary Thresholding":
            t = params.get('threshold', 127)
            gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.uint8) if img.ndim == 3 else img
            binary = np.where(gray >= t, 255, 0).astype(np.uint8)
            img = np.stack([binary]*3, axis=2) if image_data.ndim == 3 else binary

        elif op == "4. Power Law (Gamma)":
            gamma = params.get('gamma', 1.5)
            lut = np.array([np.clip(((i / 255.0) ** gamma) * 255, 0, 255) for i in range(256)], dtype=np.uint8)
            img = lut[img]

        elif op == "5. Contrast Stretching":
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)
            f = img.astype(float)
            c_min, c_max = np.min(f), np.max(f)
            if c_max != c_min:
                f = (f - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min
            img = np.clip(f, new_min, new_max).astype(image_data.dtype)

        elif op == "6. Histogram Equalization":
            if img.ndim == 2:
                img = self._equalize_channel(img)
            else:
                img = np.stack([self._equalize_channel(img[:,:,c]) for c in range(img.shape[2])], axis=2)

        elif op == "7. Horizontal Flip":
            img = np.fliplr(img)

        elif op == "8. Vertical Flip":
            img = np.flipud(img)

        elif op == "9. Salt & Pepper Noise":
            density = params.get('density', 0.05)
            out = img.copy()
            total = img.shape[0] * img.shape[1]
            num = int(total * density)
            rng = np.random.default_rng()
            salt_r = rng.integers(0, img.shape[0], num)
            salt_c = rng.integers(0, img.shape[1], num)
            out[salt_r, salt_c] = 255
            pepper_r = rng.integers(0, img.shape[0], num)
            pepper_c = rng.integers(0, img.shape[1], num)
            out[pepper_r, pepper_c] = 0
            img = out

        elif op == "10. Image Rotation":
            angle = params.get('angle', '90')
            k = {'90': 1, '180': 2, '270': 3}.get(str(angle), 1)
            img = np.rot90(img, k=k)

        return img.astype(image_data.dtype)
