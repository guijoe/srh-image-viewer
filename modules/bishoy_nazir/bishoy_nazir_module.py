from __future__ import annotations

import uuid
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
    QFormLayout,
)

from modules.i_image_module import IImageModule


# ===================== IO / Helpers ===================== #

def _read_image(file_path: str) -> np.ndarray:
    """Read image as numpy array (supports common formats via imageio or PIL)."""
    try:
        import imageio.v3 as iio  # type: ignore
        return np.asarray(iio.imread(file_path))
    except Exception:
        from PIL import Image  # type: ignore
        return np.asarray(Image.open(file_path))


def _dtype_maxv(img: np.ndarray) -> float:
    """Return a 'working max' for the input dtype."""
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    # For float images: if values look like 0..1 keep max=1, otherwise use actual max.
    img_f = img.astype(np.float32, copy=False)
    m = float(np.max(img_f)) if img_f.size else 1.0
    return m if m > 1.5 else 1.0


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False)


def _split_alpha(img_f: np.ndarray):
    if img_f.ndim == 3 and img_f.shape[2] == 4:
        return img_f[..., :3], img_f[..., 3:4]
    return img_f, None


def _recombine_alpha(rgb_f: np.ndarray, alpha):
    if alpha is None:
        return rgb_f
    return np.concatenate([rgb_f, alpha], axis=2)


def _clamp_like(img: np.ndarray, out_f: np.ndarray) -> np.ndarray:
    """Clamp to input dtype range and cast back."""
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out_f = np.clip(out_f, info.min, info.max)
        return out_f.astype(img.dtype, copy=False)
    maxv = _dtype_maxv(img)
    out_f = np.clip(out_f, 0.0, maxv)
    return out_f.astype(img.dtype, copy=False)


def _rand_from_seed(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _ensure_hwc(img_f: np.ndarray):
    """
    Normalize input to:
      - rgb: HxWxC (C=1 or 3)
      - alpha: HxWx1 or None
      - was_gray: bool
    """
    if img_f.ndim == 2:
        return img_f[..., None], None, True
    if img_f.ndim == 3:
        rgb, a = _split_alpha(img_f)
        if rgb.shape[2] == 1:
            return rgb, a, True
        return rgb, a, False
    return img_f, None, False


def _restore_from_hwc(rgb_like: np.ndarray, alpha, was_gray: bool):
    if was_gray:
        rgb_like = rgb_like[..., 0]
    return _recombine_alpha(rgb_like, alpha)


def _as_rgb3(rgb: np.ndarray) -> np.ndarray:
    """Ensure rgb is 3-channel (replicate gray channel if needed)."""
    if rgb.ndim == 3 and rgb.shape[2] == 3:
        return rgb
    if rgb.ndim == 3 and rgb.shape[2] == 1:
        return np.repeat(rgb, 3, axis=2)
    return rgb


# ===================== Effects ===================== #

def effect_negative(img_f: np.ndarray, maxv: float) -> np.ndarray:
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    out = maxv - rgb
    return _restore_from_hwc(out, alpha, was_gray)


def effect_vignette(img_f: np.ndarray, maxv: float, strength: float) -> np.ndarray:
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    h, w, _ = rgb.shape
    s = float(np.clip(strength, 0.0, 1.0))

    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    rr = np.sqrt(xx * xx + yy * yy)
    vig = 1.0 - s * np.clip((rr - 0.2) / 0.9, 0.0, 1.0) ** 2

    out = rgb * vig[..., None]
    return _restore_from_hwc(out, alpha, was_gray)


def effect_posterize_bands(img_f: np.ndarray, maxv: float, levels: int) -> np.ndarray:
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    lv = int(np.clip(levels, 2, 64))
    q = maxv / float(lv - 1)

    out = np.round(np.clip(rgb, 0.0, maxv) / q) * q
    return _restore_from_hwc(out, alpha, was_gray)


def effect_film_grain(img_f: np.ndarray, maxv: float, amount: float, seed: int) -> np.ndarray:
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    h, w, _ = rgb.shape
    a = float(np.clip(amount, 0.0, 1.0))

    rng = _rand_from_seed(seed)
    noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)

    sigma = (0.12 * a) * maxv
    out = rgb + noise * sigma
    return _restore_from_hwc(out, alpha, was_gray)


def effect_duotone(img_f: np.ndarray, maxv: float, mix: float) -> np.ndarray:
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    rgb3 = _as_rgb3(rgb)
    s = float(np.clip(mix, 0.0, 1.0))

    # Two palette endpoints (scaled by maxv)
    c0 = np.array([0.15, 0.10, 0.70], dtype=np.float32) * maxv
    c1 = np.array([0.95, 0.90, 0.20], dtype=np.float32) * maxv

    lum = (0.299 * rgb3[..., 0] + 0.587 * rgb3[..., 1] + 0.114 * rgb3[..., 2]) / maxv
    lum = np.clip(lum, 0.0, 1.0)[..., None]

    duo = (1 - lum) * c0 + lum * c1

    # Blend with original (so the knob actually “mixes”)
    out = (1.0 - s) * rgb3 + s * duo
    # Restore shape (if original was gray, return single-channel)
    if was_gray:
        out = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
    return _restore_from_hwc(out.astype(np.float32, copy=False), alpha, was_gray)


# ---- NEW CREATIVE EFFECT: Glitch RGB Split + Scanlines ---- #

def _shift_2d(channel: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Cyclic shift a 2D array."""
    return np.roll(np.roll(channel, dy, axis=0), dx, axis=1)


def effect_glitch_rgb_split(
    img_f: np.ndarray,
    maxv: float,
    intensity: float,
    seed: int,
) -> np.ndarray:
    """
    Creative effect:
      - RGB channel splitting (chromatic displacement)
      - scanlines
      - blocky horizontal/vertical jitter (seeded)
    """
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    rgb3 = _as_rgb3(rgb)

    h, w, _ = rgb3.shape
    t = float(np.clip(intensity, 0.0, 1.0))
    rng = _rand_from_seed(seed)

    # Base split amount in pixels
    max_shift = int(round(2 + 18 * t))  # 2..20 px
    dx_r = int(rng.integers(-max_shift, max_shift + 1))
    dy_r = int(rng.integers(-max_shift // 2, max_shift // 2 + 1))
    dx_b = int(rng.integers(-max_shift, max_shift + 1))
    dy_b = int(rng.integers(-max_shift // 2, max_shift // 2 + 1))

    r = _shift_2d(rgb3[..., 0], dy_r, dx_r)
    g = rgb3[..., 1]
    b = _shift_2d(rgb3[..., 2], dy_b, dx_b)

    out = np.stack([r, g, b], axis=2).astype(np.float32, copy=False)

    # Block jitter: split image into strips and shift each strip a little.
    n_strips = int(6 + 18 * t)  # 6..24
    strip_edges = np.linspace(0, h, n_strips + 1, dtype=np.int32)
    for i in range(n_strips):
        y0, y1 = int(strip_edges[i]), int(strip_edges[i + 1])
        if y1 <= y0:
            continue
        # More intensity => more jitter
        jitter = int(rng.integers(-max_shift, max_shift + 1))
        if jitter != 0:
            out[y0:y1, :, :] = np.roll(out[y0:y1, :, :], jitter, axis=1)

        # Occasional "tear": replace a thin band with a displaced band
        if t > 0.15 and rng.random() < (0.20 * t):
            band_h = int(max(1, round((y1 - y0) * 0.25)))
            by0 = int(rng.integers(y0, max(y0 + 1, y1 - band_h + 1)))
            by1 = min(y1, by0 + band_h)
            tear = np.roll(out[by0:by1, :, :], int(rng.integers(-2 * max_shift, 2 * max_shift + 1)), axis=1)
            out[by0:by1, :, :] = tear

    # Scanlines (multiplicative)
    if t > 0.0:
        yy = np.arange(h, dtype=np.float32)
        # Alternating lines, strength scales with intensity
        scan = 1.0 - (0.10 + 0.22 * t) * (0.5 + 0.5 * np.sin(2.0 * np.pi * yy / 2.0))
        out *= scan[:, None, None]

    # Tiny noise to sell the glitch
    if t > 0.0:
        noise = rng.normal(0.0, 1.0, size=(h, w, 1)).astype(np.float32)
        out += noise * ((0.03 + 0.07 * t) * maxv)

    # If original was gray, return gray (but still “glitchy” via luma)
    if was_gray:
        out_gray = (0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2])[..., None]
        out = out_gray

    return _restore_from_hwc(out, alpha, was_gray)


# ===================== UI ===================== #

class BishoyNazirControlsWidget(QWidget):
    """
    Cleaned UI:
      - Parameter labels
      - Only show relevant parameters per effect
      - Proper spin types (int for levels/seed)
    """

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # Effect selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Effect:"))
        self.operation = QComboBox()
        self.operation.addItems(
            [
                "Negative",
                "Vignette",
                "Posterize Bands",
                "Film Grain",
                "Duotone",
                "Glitch RGB Split",  # NEW
            ]
        )
        self.operation.currentTextChanged.connect(self._sync_ui)
        top.addWidget(self.operation, 1)
        root.addLayout(top)

        # Form for params
        self.form = QFormLayout()
        self.form.setLabelAlignment(Qt.AlignLeft)
        self.form.setFormAlignment(Qt.AlignTop)
        self.form.setHorizontalSpacing(12)
        self.form.setVerticalSpacing(8)
        root.addLayout(self.form)

        # Param 1 (float)
        self.p1 = QDoubleSpinBox()
        self.p1.setDecimals(3)
        self.p1.setRange(0.0, 1.0)
        self.p1.setSingleStep(0.05)
        self.p1.setValue(0.6)

        # Levels (int)
        self.levels = QSpinBox()
        self.levels.setRange(2, 64)
        self.levels.setSingleStep(1)
        self.levels.setValue(8)

        # Seed (int)
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_147_483_647)
        self.seed.setSingleStep(1)
        self.seed.setValue(42)

        # Add to form (we will hide/show rows)
        self._row_p1_label = QLabel("Strength / Mix:")
        self.form.addRow(self._row_p1_label, self.p1)

        self._row_levels_label = QLabel("Levels:")
        self.form.addRow(self._row_levels_label, self.levels)

        self._row_seed_label = QLabel("Seed:")
        self.form.addRow(self._row_seed_label, self.seed)

        # Apply
        btn = QPushButton("Apply Processing")
        btn.clicked.connect(self._apply)
        root.addWidget(btn)

        root.addStretch(1)

        self._sync_ui(self.operation.currentText())

    def _set_row_visible(self, label: QLabel, widget: QWidget, visible: bool):
        label.setVisible(visible)
        widget.setVisible(visible)

    def _sync_ui(self, op: str):
        # Hide everything by default, then enable what we need.
        self._set_row_visible(self._row_p1_label, self.p1, False)
        self._set_row_visible(self._row_levels_label, self.levels, False)
        self._set_row_visible(self._row_seed_label, self.seed, False)

        if op == "Negative":
            return

        if op == "Vignette":
            self._row_p1_label.setText("Strength (0..1):")
            self.p1.setValue(0.6)
            self._set_row_visible(self._row_p1_label, self.p1, True)
            return

        if op == "Posterize Bands":
            self._set_row_visible(self._row_levels_label, self.levels, True)
            return

        if op == "Film Grain":
            self._row_p1_label.setText("Amount (0..1):")
            self.p1.setValue(0.3)
            self._set_row_visible(self._row_p1_label, self.p1, True)
            self._set_row_visible(self._row_seed_label, self.seed, True)
            return

        if op == "Duotone":
            self._row_p1_label.setText("Mix (0..1):")
            self.p1.setValue(0.75)
            self._set_row_visible(self._row_p1_label, self.p1, True)
            return

        if op == "Glitch RGB Split":
            self._row_p1_label.setText("Intensity (0..1):")
            self.p1.setValue(0.55)
            self._set_row_visible(self._row_p1_label, self.p1, True)
            self._set_row_visible(self._row_seed_label, self.seed, True)
            return

    def _apply(self):
        op = self.operation.currentText()
        params: dict = {"operation": op}

        if op == "Vignette":
            params["strength"] = float(self.p1.value())
        elif op == "Posterize Bands":
            params["levels"] = int(self.levels.value())
        elif op == "Film Grain":
            params["amount"] = float(self.p1.value())
            params["seed"] = int(self.seed.value())
        elif op == "Duotone":
            params["mix"] = float(self.p1.value())
        elif op == "Glitch RGB Split":
            params["intensity"] = float(self.p1.value())
            params["seed"] = int(self.seed.value())

        self.module_manager.apply_processing_to_current_image(params)


# ===================== Module ===================== #

class BishoyNazirImageModule(IImageModule):

    def get_name(self) -> str:
        return "Bishoy Nazir Module"

    def get_supported_formats(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def create_control_widget(self, module_manager) -> QWidget:
        return BishoyNazirControlsWidget(module_manager=module_manager)

    def load_image(self, file_path: str):
        try:
            image_data = _read_image(file_path)
            metadata = {
                "source_path": file_path,
                "dtype": str(image_data.dtype),
                "shape": tuple(image_data.shape),
            }
            session_id = str(uuid.uuid4())
            return True, image_data, metadata, session_id
        except Exception as e:
            print(f"[BishoyNazirImageModule] load_image error: {e}")
            return False, None, {}, ""

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict | None) -> np.ndarray:
        if params is None:
            return image_data

        op = str(params.get("operation", ""))
        img_f = _to_float(image_data)
        maxv = _dtype_maxv(image_data)

        if op == "Negative":
            out_f = effect_negative(img_f, maxv)
        elif op == "Vignette":
            out_f = effect_vignette(img_f, maxv, float(params.get("strength", 0.6)))
        elif op == "Posterize Bands":
            out_f = effect_posterize_bands(img_f, maxv, int(params.get("levels", 8)))
        elif op == "Film Grain":
            out_f = effect_film_grain(
                img_f,
                maxv,
                float(params.get("amount", 0.3)),
                int(params.get("seed", 42)),
            )
        elif op == "Duotone":
            out_f = effect_duotone(img_f, maxv, float(params.get("mix", 0.75)))
        elif op == "Glitch RGB Split":
            out_f = effect_glitch_rgb_split(
                img_f,
                maxv,
                float(params.get("intensity", 0.55)),
                int(params.get("seed", 42)),
            )
        else:
            out_f = img_f

        return _clamp_like(image_data, out_f)


ImageModule = BishoyNazirImageModule