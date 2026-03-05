from __future__ import annotations

import numpy as np
import imageio

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

from skimage.restoration import denoise_bilateral
from skimage.feature import canny
from skimage.morphology import dilation, disk

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore


# ----------------------------
# Core image processing helpers
# ----------------------------

def _to_float01(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Convert an image array to float32 in [0, 1] while remembering how to convert back.
    Supports uint8/uint16/float images, grayscale or RGB/RGBA.
    """
    meta: dict = {"orig_dtype": x.dtype, "orig_min": None, "orig_max": None}

    if np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32, copy=False)
        # If float is not normalized, normalize using min/max.
        mn = float(np.nanmin(xf))
        mx = float(np.nanmax(xf))
        meta["orig_min"] = mn
        meta["orig_max"] = mx
        if mx > mn:
            xf = (xf - mn) / (mx - mn)
        else:
            xf = np.zeros_like(xf, dtype=np.float32)
        xf = np.clip(xf, 0.0, 1.0)
        return xf, meta

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        xf = x.astype(np.float32) / float(info.max)
        return xf, meta

    # Fallback: treat as float-like
    xf = x.astype(np.float32)
    mn = float(np.nanmin(xf))
    mx = float(np.nanmax(xf))
    meta["orig_min"] = mn
    meta["orig_max"] = mx
    if mx > mn:
        xf = (xf - mn) / (mx - mn)
    else:
        xf = np.zeros_like(xf, dtype=np.float32)
    xf = np.clip(xf, 0.0, 1.0)
    return xf, meta


def _from_float01(x01: np.ndarray, meta: dict) -> np.ndarray:
    """Convert float32 [0,1] back to the original dtype as best as possible."""
    x01 = np.clip(x01, 0.0, 1.0)
    orig_dtype = meta.get("orig_dtype", np.uint8)

    if np.issubdtype(orig_dtype, np.floating):
        mn = meta.get("orig_min", 0.0)
        mx = meta.get("orig_max", 1.0)
        if mx is None or mn is None or mx == mn:
            return x01.astype(orig_dtype)
        out = x01 * (mx - mn) + mn
        return out.astype(orig_dtype)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        out = np.rint(x01 * float(info.max))
        return np.clip(out, 0, info.max).astype(orig_dtype)

    return x01


def enhance_shadows_contrast_brightness(
    img: np.ndarray,
    shadows: float,
    contrast: float,
    brightness: float,
) -> np.ndarray:
    """
    Enhance:
      - shadows: lift dark regions (0.0..1.0 typical)
      - contrast: global contrast around mid-gray (0.5..2.0 typical; 1.0 = no change)
      - brightness: additive brightness shift (-1.0..1.0, but typical -0.3..0.3)

    Works for grayscale, RGB, or RGBA. Preserves alpha if present.
    """
    if img is None:
        return img

    x01, meta = _to_float01(img)

    # Separate alpha if RGBA
    alpha = None
    if x01.ndim == 3 and x01.shape[2] == 4:
        alpha = x01[..., 3:4]
        x01 = x01[..., :3]

    # ---- 1) Shadow lift (luminance mask) ----
    shadows = float(np.clip(shadows, 0.0, 1.5))

    if shadows > 0:
        if x01.ndim == 2:
            lum = x01
        else:
            # Rec.709 luminance weights
            lum = 0.2126 * x01[..., 0] + 0.7152 * x01[..., 1] + 0.0722 * x01[..., 2]

        # Mask: strongest in darks, fades in brights
        shadow_mask = np.power(1.0 - lum, 2.0)

        # Lift up to ~0.5 at max shadows=1.0 in deepest shadows (clipped later)
        lift = (0.50 * shadows) * shadow_mask

        if x01.ndim == 2:
            x01 = x01 + lift
        else:
            x01 = x01 + lift[..., None]

    # ---- 2) Brightness (additive) ----
    brightness = float(np.clip(brightness, -1.0, 1.0))
    if brightness != 0:
        x01 = x01 + brightness

    # ---- 3) Contrast (around mid gray 0.5) ----
    contrast = float(np.clip(contrast, 0.0, 4.0))
    if contrast != 1.0:
        x01 = (x01 - 0.5) * contrast + 0.5

    x01 = np.clip(x01, 0.0, 1.0)

    # Re-attach alpha if present
    if alpha is not None:
        x01 = np.concatenate([x01, np.clip(alpha, 0.0, 1.0)], axis=2)

    return _from_float01(x01, meta)


def cartoonize(
    img: np.ndarray,
    n_levels: int = 8,
    sigma_color: float = 0.08,
    sigma_spatial: float = 5.0,
    edge_sigma: float = 1.2,
    edge_threshold: float = 0.18,
    edge_thickness: int = 1,
) -> np.ndarray:
    """
    Cartoon effect:
      1) smooth colors with bilateral filter (keeps edges)
      2) posterize (reduce number of color levels)
      3) detect edges with Canny and draw them over the image

    Works for grayscale, RGB, or RGBA. Preserves alpha if present.
    """
    if img is None:
        return img

    x01, meta = _to_float01(img)

    # Separate alpha if RGBA
    alpha = None
    if x01.ndim == 3 and x01.shape[2] == 4:
        alpha = x01[..., 3:4]
        x01 = x01[..., :3]

    # ---- Smooth (bilateral filter) ----
    if x01.ndim == 3:
        smooth = denoise_bilateral(
            x01, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=-1
        )
        gray = 0.2126 * smooth[..., 0] + 0.7152 * smooth[..., 1] + 0.0722 * smooth[..., 2]
    else:
        smooth = denoise_bilateral(x01, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
        gray = smooth

    # ---- Posterize (reduce color levels) ----
    n_levels = int(np.clip(n_levels, 2, 32))
    poster = np.floor(smooth * n_levels) / float(n_levels)
    poster = np.clip(poster, 0.0, 1.0)

    # ---- Edges ----
    edge_sigma = float(np.clip(edge_sigma, 0.1, 10.0))
    edge_threshold = float(np.clip(edge_threshold, 0.0, 1.0))

    edges = canny(
        gray,
        sigma=edge_sigma,
        low_threshold=edge_threshold,
        high_threshold=min(1.0, edge_threshold * 2.0),
    )

    edge_thickness = int(np.clip(edge_thickness, 1, 10))
    if edge_thickness > 1:
        edges = dilation(edges, footprint=disk(edge_thickness))

    # Draw edges: set edge pixels to black
    if poster.ndim == 3:
        poster[edges, :] = 0.0
    else:
        poster[edges] = 0.0

    # Re-attach alpha if present
    if alpha is not None:
        poster = np.concatenate([poster, np.clip(alpha, 0.0, 1.0)], axis=2)

    return _from_float01(poster, meta)


# ----------------------------
# UI (params) + module wiring
# ----------------------------

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class ShadowsContrastBrightnessParamsWidget(BaseParamsWidget):
    """
    Parameters:
      shadows: 0..1
      contrast: 0.5..2.0
      brightness: -0.3..0.3 (shown as -30..30)
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Shadows (lift dark areas):"))
        self.shadows = QDoubleSpinBox()
        self.shadows.setRange(0.0, 1.0)
        self.shadows.setSingleStep(0.05)
        self.shadows.setValue(0.35)
        layout.addWidget(self.shadows)

        layout.addWidget(QLabel("Contrast (1.0 = no change):"))
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.5, 2.0)
        self.contrast.setSingleStep(0.05)
        self.contrast.setValue(1.15)
        layout.addWidget(self.contrast)

        layout.addWidget(QLabel("Brightness (-30..30):"))
        self.brightness = QDoubleSpinBox()
        self.brightness.setRange(-30.0, 30.0)
        self.brightness.setSingleStep(1.0)
        self.brightness.setValue(8.0)
        layout.addWidget(self.brightness)

        layout.addStretch()

    def get_params(self) -> dict:
        # Map brightness from [-30..30] -> [-0.30..0.30]
        return {
            "shadows": float(self.shadows.value()),
            "contrast": float(self.contrast.value()),
            "brightness": float(self.brightness.value()) / 100.0,
        }


class CartoonParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Cartoon Levels (fewer = more cartoon):"))
        self.levels = QDoubleSpinBox()
        self.levels.setRange(2, 32)
        self.levels.setSingleStep(1)
        self.levels.setValue(8)
        self.levels.setDecimals(0)
        layout.addWidget(self.levels)

        layout.addWidget(QLabel("Color smoothing (sigma_color):"))
        self.sigma_color = QDoubleSpinBox()
        self.sigma_color.setRange(0.01, 0.30)
        self.sigma_color.setSingleStep(0.01)
        self.sigma_color.setValue(0.08)
        layout.addWidget(self.sigma_color)

        layout.addWidget(QLabel("Spatial smoothing (sigma_spatial):"))
        self.sigma_spatial = QDoubleSpinBox()
        self.sigma_spatial.setRange(1.0, 20.0)
        self.sigma_spatial.setSingleStep(1.0)
        self.sigma_spatial.setValue(5.0)
        layout.addWidget(self.sigma_spatial)

        layout.addWidget(QLabel("Edge softness (edge_sigma):"))
        self.edge_sigma = QDoubleSpinBox()
        self.edge_sigma.setRange(0.5, 5.0)
        self.edge_sigma.setSingleStep(0.1)
        self.edge_sigma.setValue(1.2)
        layout.addWidget(self.edge_sigma)

        layout.addWidget(QLabel("Edge sensitivity (edge_threshold):"))
        self.edge_threshold = QDoubleSpinBox()
        self.edge_threshold.setRange(0.01, 0.50)
        self.edge_threshold.setSingleStep(0.01)
        self.edge_threshold.setValue(0.18)
        layout.addWidget(self.edge_threshold)

        layout.addWidget(QLabel("Edge thickness (1–5):"))
        self.edge_thickness = QDoubleSpinBox()
        self.edge_thickness.setRange(1, 5)
        self.edge_thickness.setSingleStep(1)
        self.edge_thickness.setValue(1)
        self.edge_thickness.setDecimals(0)
        layout.addWidget(self.edge_thickness)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            "n_levels": int(self.levels.value()),
            "sigma_color": float(self.sigma_color.value()),
            "sigma_spatial": float(self.sigma_spatial.value()),
            "edge_sigma": float(self.edge_sigma.value()),
            "edge_threshold": float(self.edge_threshold.value()),
            "edge_thickness": int(self.edge_thickness.value()),
        }


class ShadowEnhancerControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets: dict[str, BaseParamsWidget] = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Shadow Enhancer</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Shadows + Contrast + Brightness": ShadowsContrastBrightnessParamsWidget,
            "Cartoon": CartoonParamsWidget,
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

        # Ensure correct widget shows initially
        if self.operation_selector.count() > 0:
            self._on_operation_changed(self.operation_selector.currentText())

    def _on_apply_clicked(self):
        op = self.operation_selector.currentText()
        params = self.param_widgets[op].get_params()
        params["operation"] = op
        self.process_requested.emit(params)

    def _on_operation_changed(self, op: str):
        if op in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[op])


class ShadowEnhancerImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Shadow Enhancer"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = ShadowEnhancerControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        """
        Mirrors the sample module approach: load via imageio and store into ImageDataStore.
        """
        try:
            image_data = imageio.imread(file_path)

            metadata = {
                "file_path": file_path,
                "shape": tuple(image_data.shape),
                "dtype": str(image_data.dtype),
            }

            session_id = ImageDataStore().set_image(image_data, metadata, session_id=None)
            return True, image_data, metadata, session_id

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = (params or {}).get("operation", "")

        if op == "Shadows + Contrast + Brightness":
            shadows = float(params.get("shadows", 0.35))
            contrast = float(params.get("contrast", 1.15))
            brightness = float(params.get("brightness", 0.08))
            return enhance_shadows_contrast_brightness(image_data, shadows, contrast, brightness)

        if op == "Cartoon":
            return cartoonize(
                image_data,
                n_levels=int(params.get("n_levels", 8)),
                sigma_color=float(params.get("sigma_color", 0.08)),
                sigma_spatial=float(params.get("sigma_spatial", 5.0)),
                edge_sigma=float(params.get("edge_sigma", 1.2)),
                edge_threshold=float(params.get("edge_threshold", 0.18)),
                edge_thickness=int(params.get("edge_thickness", 1)),
            )

        return image_data