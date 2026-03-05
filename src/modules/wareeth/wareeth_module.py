from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QStackedWidget,
    QDoubleSpinBox,
)
from PySide6.QtCore import Signal

import numpy as np
import imageio.v2 as imageio

import skimage.filters
import skimage.morphology
import skimage.exposure
from skimage.color import rgb2gray

from modules.i_image_module import IImageModule


# -------------------------
# Parameter Widgets
# -------------------------
class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """Placeholder widget for operations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class ContrastStretchingParamsWidget(BaseParamsWidget):
    """Contrast stretching params."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("New Minimum Intensity:"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(65535.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        layout.addWidget(QLabel("New Maximum Intensity:"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(65535.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"new_min": float(self.min_spinbox.value()), "new_max": float(self.max_spinbox.value())}


class BrightnessContrastParamsWidget(BaseParamsWidget):
    """Brightness/Contrast params."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness (add):"))
        self.brightness = QDoubleSpinBox()
        self.brightness.setMinimum(-255.0)
        self.brightness.setMaximum(255.0)
        self.brightness.setValue(0.0)
        self.brightness.setSingleStep(5.0)
        layout.addWidget(self.brightness)

        layout.addWidget(QLabel("Contrast (multiply):"))
        self.contrast = QDoubleSpinBox()
        self.contrast.setMinimum(0.0)
        self.contrast.setMaximum(3.0)
        self.contrast.setValue(1.0)
        self.contrast.setSingleStep(0.05)
        layout.addWidget(self.contrast)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"brightness": float(self.brightness.value()), "contrast": float(self.contrast.value())}


class UnsharpMaskParamsWidget(BaseParamsWidget):
    """Unsharp mask params."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Radius:"))
        self.radius = QDoubleSpinBox()
        self.radius.setMinimum(0.5)
        self.radius.setMaximum(10.0)
        self.radius.setValue(1.0)
        self.radius.setSingleStep(0.5)
        layout.addWidget(self.radius)

        layout.addWidget(QLabel("Amount:"))
        self.amount = QDoubleSpinBox()
        self.amount.setMinimum(0.0)
        self.amount.setMaximum(5.0)
        self.amount.setValue(1.0)
        self.amount.setSingleStep(0.1)
        layout.addWidget(self.amount)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"radius": float(self.radius.value()), "amount": float(self.amount.value())}


class CLAHEParamsWidget(BaseParamsWidget):
    """CLAHE params."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Clip limit (0.01–0.2 typical):"))
        self.clip_limit = QDoubleSpinBox()
        self.clip_limit.setMinimum(0.001)
        self.clip_limit.setMaximum(1.0)
        self.clip_limit.setValue(0.03)
        self.clip_limit.setSingleStep(0.01)
        layout.addWidget(self.clip_limit)

        layout.addWidget(QLabel("Kernel size (tile size):"))
        self.kernel_size = QDoubleSpinBox()
        self.kernel_size.setMinimum(2.0)
        self.kernel_size.setMaximum(128.0)
        self.kernel_size.setValue(8.0)
        self.kernel_size.setSingleStep(1.0)
        layout.addWidget(self.kernel_size)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"clip_limit": float(self.clip_limit.value()), "kernel_size": int(self.kernel_size.value())}


class MedianFilterParamsWidget(BaseParamsWidget):
    """Median filter params."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Radius (disk):"))
        self.radius = QDoubleSpinBox()
        self.radius.setMinimum(1.0)
        self.radius.setMaximum(25.0)
        self.radius.setValue(2.0)
        self.radius.setSingleStep(1.0)
        layout.addWidget(self.radius)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"radius": int(self.radius.value())}


class GrainParamsWidget(BaseParamsWidget):
    """Film grain / noise parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Grain strength (std dev in intensity units):"))
        self.std_spinbox = QDoubleSpinBox()
        self.std_spinbox.setMinimum(0.0)
        self.std_spinbox.setMaximum(100.0)
        self.std_spinbox.setValue(8.0)
        self.std_spinbox.setSingleStep(1.0)
        layout.addWidget(self.std_spinbox)

        layout.addWidget(QLabel("Seed (0 = random):"))
        self.seed_spinbox = QDoubleSpinBox()
        self.seed_spinbox.setMinimum(0.0)
        self.seed_spinbox.setMaximum(999999.0)
        self.seed_spinbox.setValue(0.0)
        self.seed_spinbox.setSingleStep(1.0)
        self.seed_spinbox.setDecimals(0)
        layout.addWidget(self.seed_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {"std": float(self.std_spinbox.value()), "seed": int(self.seed_spinbox.value())}


# -------------------------
# Controls Widget
# -------------------------
class WareethControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Wareeth Control Panel</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # (internal_key, display_label, widget_class)
        ops = [
            ("CONTRAST_STRETCH", "Contrast Stretching / Dynamic Range Expansion", ContrastStretchingParamsWidget),
            ("BRIGHT_CONTRAST", "Brightness / Contrast / Light & Punch Control", BrightnessContrastParamsWidget),
            ("INVERT", "Invert Colors / Photo Negative Mode", NoParamsWidget),

            ("UNSHARP", "Sharpen (Unsharp Mask) / Detail Booster", UnsharpMaskParamsWidget),

            ("HISTEQ", "Histogram Equalization / Auto Contrast Boost", NoParamsWidget),
            ("CLAHE", "CLAHE / Smart Local Contrast Boost", CLAHEParamsWidget),

            ("MEDIAN", "Median Filter / Noise Cleanup Filter", MedianFilterParamsWidget),

            ("OTSU", "Otsu Thresholding / Auto Black & White Mode", NoParamsWidget),

            ("GRAIN", "Film Grain / Grain Filter", GrainParamsWidget),
        ]

        for key, label, widget_cls in ops:
            widget = widget_cls()
            self.params_stack.addWidget(widget)
            self.operation_selector.addItem(label, key)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.operation_selector.currentIndexChanged.connect(self.params_stack.setCurrentIndex)
        self.apply_button.clicked.connect(self._on_apply_clicked)

        if self.operation_selector.count() > 0:
            self.params_stack.setCurrentIndex(0)

    def _on_apply_clicked(self):
        idx = self.operation_selector.currentIndex()
        op_key = self.operation_selector.itemData(idx)
        widget = self.params_stack.currentWidget()

        params = widget.get_params() if hasattr(widget, "get_params") else {}
        params["operation"] = op_key

        self.process_requested.emit(params)


# -------------------------
# Module
# -------------------------
class WareethImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Wareeth Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = WareethControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {"name": file_path.split("/")[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    # ---------- Helpers ----------
    @staticmethod
    def _dtype_min_max(dtype):
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        return 0.0, 1.0

    @staticmethod
    def _is_rgb_or_rgba(img: np.ndarray) -> bool:
        return img.ndim == 3 and img.shape[-1] in (3, 4)

    @staticmethod
    def _split_rgb_alpha(img: np.ndarray):
        if img.ndim == 3 and img.shape[-1] == 4:
            return img[..., :3], img[..., 3:4]
        return img, None

    @staticmethod
    def _merge_rgb_alpha(rgb: np.ndarray, alpha):
        if alpha is None:
            return rgb
        return np.concatenate([rgb, alpha], axis=-1)

    @staticmethod
    def _to_grayscale(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            return rgb2gray(img[..., :3])
        return img

    @staticmethod
    def _normalize_to_unit(img: np.ndarray) -> tuple[np.ndarray, float, float]:
        imgf = img.astype(np.float32)
        imin = float(np.min(imgf)) if imgf.size else 0.0
        imax = float(np.max(imgf)) if imgf.size else 1.0
        if imax == imin:
            return np.zeros_like(imgf, dtype=np.float32), imin, imax
        unit = (imgf - imin) / (imax - imin)
        unit = np.clip(unit, 0.0, 1.0)
        return unit, imin, imax

    @staticmethod
    def _denormalize_from_unit(unit: np.ndarray, imin: float, imax: float) -> np.ndarray:
        return unit.astype(np.float32) * (imax - imin) + imin

    # ---------- Processing ----------
    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        op = params.get("operation")

        # ---- Intensity transforms ----
        if op == "CONTRAST_STRETCH":
            img_float = processed_data.astype(float)
            new_min = float(params.get("new_min", 0.0))
            new_max = float(params.get("new_max", 255.0))
            current_min = float(np.min(img_float))
            current_max = float(np.max(img_float))

            if current_max != current_min:
                stretched = (img_float - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min
                processed_data = np.clip(stretched, new_min, new_max)

        elif op == "BRIGHT_CONTRAST":
            b = float(params.get("brightness", 0.0))
            c = float(params.get("contrast", 1.0))
            processed_data = processed_data.astype(float) * c + b

        elif op == "INVERT":
            img_float = processed_data.astype(float)
            if np.issubdtype(image_data.dtype, np.integer):
                dmin, dmax = self._dtype_min_max(image_data.dtype)
                processed_data = (dmax + dmin) - img_float
            else:
                fmin = float(np.min(img_float))
                fmax = float(np.max(img_float))
                processed_data = (fmax + fmin) - img_float

        # ---- Sharpen ----
        elif op == "UNSHARP":
            radius = float(params.get("radius", 1.0))
            amount = float(params.get("amount", 1.0))
            processed_data = skimage.filters.unsharp_mask(
                processed_data.astype(float),
                radius=radius,
                amount=amount,
                preserve_range=True,
            )

        # ---- Histogram enhancement ----
        elif op == "HISTEQ":
            img = processed_data
            if self._is_rgb_or_rgba(img):
                rgb, alpha = self._split_rgb_alpha(img)
                unit, imin, imax = self._normalize_to_unit(rgb)
                out = np.empty_like(unit, dtype=np.float32)
                for ch in range(unit.shape[-1]):
                    out[..., ch] = skimage.exposure.equalize_hist(unit[..., ch]).astype(np.float32)
                rgb_out = self._denormalize_from_unit(out, imin, imax)
                processed_data = self._merge_rgb_alpha(rgb_out, alpha)
            else:
                unit, imin, imax = self._normalize_to_unit(img)
                out = skimage.exposure.equalize_hist(unit).astype(np.float32)
                processed_data = self._denormalize_from_unit(out, imin, imax)

        elif op == "CLAHE":
            clip_limit = float(params.get("clip_limit", 0.03))
            kernel_size = int(params.get("kernel_size", 8))
            img = processed_data

            if self._is_rgb_or_rgba(img):
                rgb, alpha = self._split_rgb_alpha(img)
                unit, imin, imax = self._normalize_to_unit(rgb)
                out = np.empty_like(unit, dtype=np.float32)
                for ch in range(unit.shape[-1]):
                    out[..., ch] = skimage.exposure.equalize_adapthist(
                        unit[..., ch],
                        clip_limit=clip_limit,
                        kernel_size=kernel_size,
                    ).astype(np.float32)
                rgb_out = self._denormalize_from_unit(out, imin, imax)
                processed_data = self._merge_rgb_alpha(rgb_out, alpha)
            else:
                unit, imin, imax = self._normalize_to_unit(img)
                out = skimage.exposure.equalize_adapthist(
                    unit,
                    clip_limit=clip_limit,
                    kernel_size=kernel_size,
                ).astype(np.float32)
                processed_data = self._denormalize_from_unit(out, imin, imax)

        # ---- Median filter ----
        elif op == "MEDIAN":
            radius = int(params.get("radius", 2))
            footprint = skimage.morphology.disk(radius)
            img = processed_data

            if self._is_rgb_or_rgba(img):
                rgb, alpha = self._split_rgb_alpha(img)
                out_channels = [
                    skimage.filters.median(rgb[..., ch], footprint=footprint)
                    for ch in range(rgb.shape[-1])
                ]
                rgb_out = np.stack(out_channels, axis=-1)
                processed_data = self._merge_rgb_alpha(rgb_out, alpha)
            else:
                processed_data = skimage.filters.median(img, footprint=footprint)

        # ---- Otsu ----
        elif op == "OTSU":
            gray = self._to_grayscale(processed_data).astype(float)
            try:
                t = float(skimage.filters.threshold_otsu(gray))
            except Exception:
                t = float(np.mean(gray)) if gray.size else 0.0

            mask = gray >= t
            if np.issubdtype(image_data.dtype, np.integer):
                _, dmax = self._dtype_min_max(image_data.dtype)
                processed_data = mask.astype(float) * dmax
            else:
                processed_data = mask.astype(float)

        # ---- Film grain ----
        elif op == "GRAIN":
            std = float(params.get("std", 8.0))
            seed = int(params.get("seed", 0))
            rng = np.random.default_rng(None if seed == 0 else seed)

            img_float = processed_data.astype(float)
            noise = rng.normal(loc=0.0, scale=std, size=img_float.shape)
            processed_data = img_float + noise

        # ---- Clip + cast back ----
        if np.issubdtype(image_data.dtype, np.integer):
            dmin, dmax = self._dtype_min_max(image_data.dtype)
            processed_data = np.clip(processed_data, dmin, dmax).astype(image_data.dtype)
        else:
            processed_data = processed_data.astype(np.float32)

        return processed_data