from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QSpinBox
from PySide6.QtCore import Signal
import numpy as np

from modules.i_image_module import IImageModule


# ─── Parameter Widgets ────────────────────────────────────────────────────────

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        lbl = QLabel("No parameters needed.")
        lbl.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(lbl)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class BrightnessParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Brightness offset (-255 to 255):"))
        self.spin = QSpinBox()
        self.spin.setMinimum(-255)
        self.spin.setMaximum(255)
        self.spin.setValue(50)
        layout.addWidget(self.spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'offset': self.spin.value()}

class ContrastParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Contrast factor (0.1 – 5.0):"))
        self.spin = QDoubleSpinBox()
        self.spin.setMinimum(0.1)
        self.spin.setMaximum(5.0)
        self.spin.setSingleStep(0.1)
        self.spin.setValue(1.5)
        layout.addWidget(self.spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'factor': self.spin.value()}

class SharpenStrengthWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sharpening strength (0.1 – 5.0):"))
        self.spin = QDoubleSpinBox()
        self.spin.setMinimum(0.1)
        self.spin.setMaximum(5.0)
        self.spin.setSingleStep(0.1)
        self.spin.setValue(1.0)
        layout.addWidget(self.spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'strength': self.spin.value()}

class BlurParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel size (odd number, 3–21):"))
        self.spin = QSpinBox()
        self.spin.setMinimum(3)
        self.spin.setMaximum(21)
        self.spin.setSingleStep(2)
        self.spin.setValue(5)
        layout.addWidget(self.spin)
        layout.addStretch()

    def get_params(self) -> dict:
        k = self.spin.value()
        if k % 2 == 0:
            k += 1  # ensure odd
        return {'kernel_size': k}


# ─── Control Widget ───────────────────────────────────────────────────────────

class MyControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>My Module</h3>"))
        layout.addWidget(QLabel("Operation:"))

        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Brightness Adjustment": BrightnessParamsWidget,
            "Contrast Stretch": ContrastParamsWidget,
            "Histogram Equalization": NoParamsWidget,
            "Sharpen (Unsharp Mask)": SharpenStrengthWidget,
            "Box Blur": BlurParamsWidget,
            "Grayscale": NoParamsWidget,
            "Flip Horizontal": NoParamsWidget,
            "Flip Vertical": NoParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply(self):
        name = self.operation_selector.currentText()
        params = self.param_widgets[name].get_params()
        params['operation'] = name
        self.process_requested.emit(params)

    def _on_operation_changed(self, name: str):
        if name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[name])


# ─── Module ───────────────────────────────────────────────────────────────────

class AlexModule(IImageModule):
    """
    A simple image processing module implementing eight transformations
    using NumPy only (no skimage/scipy for the algorithms themselves).
    """

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Alex Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = MyControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_request)
        return self._controls_widget

    def _handle_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            import imageio
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _box_blur_2d(channel: np.ndarray, k: int) -> np.ndarray:
        """Pure-NumPy box blur via two 1-D passes (separable filter)."""
        pad = k // 2
        # Horizontal pass
        padded = np.pad(channel.astype(float), ((0, 0), (pad, pad)), mode='edge')
        out = np.zeros_like(channel, dtype=float)
        for i in range(k):
            out += padded[:, i:i + channel.shape[1]]
        out /= k
        # Vertical pass
        padded2 = np.pad(out, ((pad, pad), (0, 0)), mode='edge')
        result = np.zeros_like(channel, dtype=float)
        for i in range(k):
            result += padded2[i:i + channel.shape[0], :]
        result /= k
        return result

    @staticmethod
    def _apply_per_channel(image: np.ndarray, func) -> np.ndarray:
        """Apply func to each channel of an RGB/RGBA image."""
        channels = [func(image[:, :, c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)

    # ── Process ──────────────────────────────────────────────────────────────

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get('operation', '')
        img = image_data.copy().astype(np.float64)
        original_dtype = image_data.dtype

        # Determine whether image has alpha channel
        has_alpha = img.ndim == 3 and img.shape[2] == 4
        rgb_channels = slice(0, 3) if has_alpha else slice(0, img.shape[2] if img.ndim == 3 else 1)

        # ── 1. Brightness Adjustment ─────────────────────────────────────────
        # Simply add a constant to every pixel value.
        # Brighter pixel = higher number; we clip so values stay in [0, 255].
        if op == "Brightness Adjustment":
            offset = params.get('offset', 50)
            result = np.clip(img + offset, 0, 255)

        # ── 2. Contrast Stretch ──────────────────────────────────────────────
        # Multiply the distance of each pixel from the mean by a factor.
        # factor > 1 → more contrast; factor < 1 → less contrast.
        elif op == "Contrast Stretch":
            factor = params.get('factor', 1.5)
            mean = np.mean(img)
            result = np.clip(mean + factor * (img - mean), 0, 255)

        # ── 3. Histogram Equalization ────────────────────────────────────────
        # Redistributes pixel intensities so the histogram is flat.
        # Works on luminance to avoid shifting hues.
        elif op == "Histogram Equalization":
            if img.ndim == 2:
                # Grayscale: equalize directly
                flat = img.flatten().astype(int)
                hist, _ = np.histogram(flat, bins=256, range=(0, 256))
                cdf = hist.cumsum()
                cdf_min = cdf[cdf > 0].min()
                total_pixels = img.size
                lut = np.round(
                    (cdf - cdf_min) / (total_pixels - cdf_min) * 255
                ).astype(np.uint8)
                result = lut[img.astype(int)].astype(float)
            else:
                # Color: convert to YCbCr, equalize Y only
                R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                Y  =  0.299 * R + 0.587 * G + 0.114 * B
                Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128
                Cr =  0.5 * R - 0.418688 * G - 0.081312 * B + 128

                flat = Y.flatten().astype(int).clip(0, 255)
                hist, _ = np.histogram(flat, bins=256, range=(0, 256))
                cdf = hist.cumsum()
                cdf_min = cdf[cdf > 0].min()
                total_pixels = Y.size
                lut = np.round(
                    (cdf - cdf_min) / (total_pixels - cdf_min) * 255
                ).clip(0, 255).astype(np.uint8)
                Y_eq = lut[Y.astype(int).clip(0, 255)].astype(float)

                # Convert back to RGB
                new_R = Y_eq + 1.402 * (Cr - 128)
                new_G = Y_eq - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128)
                new_B = Y_eq + 1.772 * (Cb - 128)
                result = img.copy()
                result[:, :, 0] = np.clip(new_R, 0, 255)
                result[:, :, 1] = np.clip(new_G, 0, 255)
                result[:, :, 2] = np.clip(new_B, 0, 255)

        # ── 4. Sharpen (Unsharp Mask) ────────────────────────────────────────
        # Sharpened = original + strength × (original − blurred)
        # The "unsharp mask" is the difference between original and blurred.
        elif op == "Sharpen (Unsharp Mask)":
            strength = params.get('strength', 1.0)
            k = 5  # fixed small blur kernel size for the mask

            def sharpen_channel(ch):
                blurred = self._box_blur_2d(ch, k)
                mask = ch - blurred
                return np.clip(ch + strength * mask, 0, 255)

            if img.ndim == 2:
                result = sharpen_channel(img)
            else:
                result = img.copy()
                for c in range(3):
                    result[:, :, c] = sharpen_channel(img[:, :, c])

        # ── 5. Box Blur ──────────────────────────────────────────────────────
        # Replaces each pixel with the average of its k×k neighbourhood.
        # Simple but effective smoothing / noise reduction.
        elif op == "Box Blur":
            k = params.get('kernel_size', 5)
            if img.ndim == 2:
                result = self._box_blur_2d(img, k)
            else:
                result = img.copy()
                for c in range(3):
                    result[:, :, c] = self._box_blur_2d(img[:, :, c], k)

        # ── 6. Grayscale ─────────────────────────────────────────────────────
        # Weighted sum of R, G, B matching human eye sensitivity.
        elif op == "Grayscale":
            if img.ndim == 3:
                gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
                result = np.stack([gray, gray, gray], axis=-1)
                if has_alpha:
                    result = np.concatenate([result, img[:, :, 3:4]], axis=-1)
            else:
                result = img

        # ── 7. Flip Horizontal ───────────────────────────────────────────────
        elif op == "Flip Horizontal":
            result = img[:, ::-1]

        # ── 8. Flip Vertical ─────────────────────────────────────────────────
        elif op == "Flip Vertical":
            result = img[::-1, :]

        else:
            result = img

        return np.clip(result, 0, 255).astype(original_dtype)
