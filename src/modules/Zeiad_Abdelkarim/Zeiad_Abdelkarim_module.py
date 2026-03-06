from __future__ import annotations
import uuid
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QPushButton
)

from modules.i_image_module import IImageModule


def _read_image(file_path: str) -> np.ndarray:
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(file_path))
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(file_path))


def _dtype_maxv(img: np.ndarray) -> float:
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    img_f = img.astype(np.float32)
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
    if np.issubdtype(img.dtype, np.integer):
        info = np.iinfo(img.dtype)
        out_f = np.clip(out_f, info.min, info.max)
        return out_f.astype(img.dtype)
    maxv = _dtype_maxv(img)
    out_f = np.clip(out_f, 0.0, maxv)
    return out_f.astype(img.dtype)


def _rand_from_seed(seed: int):
    return np.random.default_rng(int(seed) & 0xFFFFFFFF)


def _ensure_hwc(img_f: np.ndarray):
    
    if img_f.ndim == 2:
        return img_f[..., None], None, True
    if img_f.ndim == 3:
        rgb, a = _split_alpha(img_f)
        return rgb, a, False
    return img_f, None, False


def _restore_from_hwc(rgb_like: np.ndarray, alpha, was_gray: bool):
    if was_gray:
        rgb_like = rgb_like[..., 0]
    return _recombine_alpha(rgb_like, alpha)

def _box_blur(rgb: np.ndarray, k: int) -> np.ndarray:
    """Fast-ish box blur using integral image. rgb is HxWxC float32."""
    k = int(k)
    if k <= 1:
        return rgb
    if k % 2 == 0:
        k += 1
    r = k // 2

    h, w, c = rgb.shape
    pad = ((r, r), (r, r), (0, 0))
    x = np.pad(rgb, pad, mode="edge")


    ii = x.cumsum(axis=0).cumsum(axis=1)


    y0, y1 = 0, h
    x0, x1 = 0, w

    A = ii[y0:y1, x0:x1]
    B = ii[y0:y1, x0 + k:x1 + k]
    C = ii[y0 + k:y1 + k, x0:x1]
    D = ii[y0 + k:y1 + k, x0 + k:x1 + k]

    s = D - C - B + A
    return s / float(k * k)


def effect_sepia(img_f: np.ndarray, maxv: float, intensity: float) -> np.ndarray:
    """Sepia tone (intensity 0..1)."""
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    t = float(np.clip(intensity, 0.0, 1.0))

    if rgb.shape[2] == 1:
        rgb3 = np.repeat(rgb, 3, axis=2)
    else:
        rgb3 = rgb[..., :3]

    # normalize to 0..1
    x = np.clip(rgb3 / maxv, 0.0, 1.0)

    # sepia matrix
    sep = np.empty_like(x)
    sep[..., 0] = 0.393 * x[..., 0] + 0.769 * x[..., 1] + 0.189 * x[..., 2]
    sep[..., 1] = 0.349 * x[..., 0] + 0.686 * x[..., 1] + 0.168 * x[..., 2]
    sep[..., 2] = 0.272 * x[..., 0] + 0.534 * x[..., 1] + 0.131 * x[..., 2]
    sep = np.clip(sep, 0.0, 1.0)

    out3 = (1.0 - t) * x + t * sep
    out3 = out3 * maxv

    if rgb.shape[2] == 1:
        out = (0.299 * out3[..., 0] + 0.587 * out3[..., 1] + 0.114 * out3[..., 2])[..., None]
    else:
        out = out3

    return _restore_from_hwc(out.astype(np.float32), alpha, was_gray)


def effect_soft_blur(img_f: np.ndarray, maxv: float, strength: float, ksize: int) -> np.ndarray:
    """Soft blur (strength 0..1), kernel size from ksize (odd)."""
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    s = float(np.clip(strength, 0.0, 1.0))
    k = int(np.clip(ksize, 1, 51))

    if rgb.shape[2] == 1:
        blurred = _box_blur(rgb, k)
        out = (1.0 - s) * rgb + s * blurred
    else:
        blurred = _box_blur(rgb[..., :3], k)
        base = rgb[..., :3]
        out3 = (1.0 - s) * base + s * blurred

        if rgb.shape[2] > 3:
            tail = rgb[..., 3:]
            out = np.concatenate([out3, tail], axis=2)
        else:
            out = out3

    return _restore_from_hwc(out.astype(np.float32), alpha, was_gray)


def effect_edge_sobel(img_f: np.ndarray, maxv: float, threshold: float) -> np.ndarray:
    """Sobel edge highlight (threshold 0..1). Outputs edge map (grayscale) or applied to RGB."""
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    th = float(np.clip(threshold, 0.0, 1.0))

    if rgb.shape[2] == 1:
        lum = rgb[..., 0]
    else:
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])

   
    x = np.pad(lum, ((1, 1), (1, 1)), mode="edge")

   
    gx = (
        -1 * x[:-2, :-2] + 1 * x[:-2, 2:] +
        -2 * x[1:-1, :-2] + 2 * x[1:-1, 2:] +
        -1 * x[2:, :-2] + 1 * x[2:, 2:]
    )
    gy = (
        -1 * x[:-2, :-2] + -2 * x[:-2, 1:-1] + -1 * x[:-2, 2:] +
         1 * x[2:, :-2] +  2 * x[2:, 1:-1] +  1 * x[2:, 2:]
    )

    mag = np.sqrt(gx * gx + gy * gy)

    
    mag_n = mag / (mag.max() + 1e-6)
    edges = (mag_n > th).astype(np.float32) * maxv

    out = edges[..., None]
    return _restore_from_hwc(out.astype(np.float32), alpha, was_gray)


def effect_pixelate(img_f: np.ndarray, maxv: float, block_size: int) -> np.ndarray:
    """Pixelation by block averaging. block_size >= 1."""
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    b = int(np.clip(block_size, 1, 128))
    h, w, c = rgb.shape

    hh = (h // b) * b
    ww = (w // b) * b

    core = rgb[:hh, :ww]
    core = core.reshape(hh // b, b, ww // b, b, c).mean(axis=(1, 3))
    core = np.repeat(np.repeat(core, b, axis=0), b, axis=1)

    out = rgb.copy()
    out[:hh, :ww] = core

    return _restore_from_hwc(out.astype(np.float32), alpha, was_gray)


def _rgb_to_hsv01(rgb01: np.ndarray) -> np.ndarray:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn

    h = np.zeros_like(mx)
    s = np.zeros_like(mx)
    v = mx

    s = np.where(mx == 0, 0, d / (mx + 1e-12))

    mask = d > 1e-12
    rc = np.where(mask, (mx - r) / (d + 1e-12), 0)
    gc = np.where(mask, (mx - g) / (d + 1e-12), 0)
    bc = np.where(mask, (mx - b) / (d + 1e-12), 0)

    h = np.where((mx == r) & mask, (bc - gc), h)
    h = np.where((mx == g) & mask, 2.0 + (rc - bc), h)
    h = np.where((mx == b) & mask, 4.0 + (gc - rc), h)
    h = (h / 6.0) % 1.0

    return np.stack([h, s, v], axis=-1).astype(np.float32)


def _hsv01_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6.0).astype(np.int32)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)

    i_mod = i % 6
    r = np.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [v, q, p, p, t, v],
        default=v
    )
    g = np.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [t, v, v, q, p, p],
        default=v
    )
    b = np.select(
        [i_mod == 0, i_mod == 1, i_mod == 2, i_mod == 3, i_mod == 4, i_mod == 5],
        [p, p, t, v, v, q],
        default=v
    )
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def effect_hue_shift(img_f: np.ndarray, maxv: float, shift01: float) -> np.ndarray:
    """Hue shift where shift01 is 0..1 => 0..360deg."""
    rgb, alpha, was_gray = _ensure_hwc(img_f)
    sh = float(np.clip(shift01, 0.0, 1.0))

    if rgb.shape[2] == 1:
        return _restore_from_hwc(rgb.astype(np.float32), alpha, was_gray)

    x = np.clip(rgb[..., :3] / maxv, 0.0, 1.0)
    hsv = _rgb_to_hsv01(x)
    hsv[..., 0] = (hsv[..., 0] + sh) % 1.0
    out3 = _hsv01_to_rgb(hsv) * maxv

    if rgb.shape[2] > 3:
        out = np.concatenate([out3, rgb[..., 3:]], axis=2)
    else:
        out = out3

    return _restore_from_hwc(out.astype(np.float32), alpha, was_gray)




class ZeiadAbdelkarimControlsWidget(QWidget):

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Effect:"))

        self.operation = QComboBox()
        self.operation.addItems([
            "Sepia Tone",
            "Soft Blur",
            "Sobel Edges",
            "Pixelate",
            "Hue Shift",
        ])
        row.addWidget(self.operation)
        layout.addLayout(row)

       
        self.p1 = QDoubleSpinBox()
        self.p1.setRange(0.0, 1.0)
        self.p1.setSingleStep(0.05)
        self.p1.setValue(0.5)

       
        self.p2 = QDoubleSpinBox()
        self.p2.setRange(1, 51)
        self.p2.setDecimals(0)
        self.p2.setSingleStep(2)
        self.p2.setValue(9)

        layout.addWidget(QLabel("Param 1 (0..1):"))
        layout.addWidget(self.p1)
        layout.addWidget(QLabel("Param 2 (size):"))
        layout.addWidget(self.p2)

        btn = QPushButton("Apply Processing")
        btn.clicked.connect(self._apply)
        layout.addWidget(btn)

        layout.addStretch()

    def _apply(self):
        op = self.operation.currentText()
        params = {"operation": op}

        if op == "Sepia Tone":
            params["intensity"] = float(self.p1.value())
        elif op == "Soft Blur":
            params["strength"] = float(self.p1.value())
            params["ksize"] = int(self.p2.value())
        elif op == "Sobel Edges":
            params["threshold"] = float(self.p1.value())
        elif op == "Pixelate":
            params["block"] = int(self.p2.value())
        elif op == "Hue Shift":
            params["shift"] = float(self.p1.value())

        self.module_manager.apply_processing_to_current_image(params)




class ZeiadAbdelkarimImageModule(IImageModule):

    def get_name(self) -> str:
        return "Zeiad Abdelkarim Module"

    def get_supported_formats(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    def create_control_widget(self, module_manager) -> QWidget:
        return ZeiadAbdelkarimControlsWidget(module_manager=module_manager)

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
            print(f"[ZeiadAbdelkarimImageModule] load_image error: {e}")
            return False, None, {}, ""

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict | None) -> np.ndarray:
        if params is None:
            return image_data

        op = params.get("operation", "")
        img_f = _to_float(image_data)
        maxv = _dtype_maxv(image_data)

        if op == "Sepia Tone":
            out_f = effect_sepia(img_f, maxv, params.get("intensity", 0.7))
        elif op == "Soft Blur":
            out_f = effect_soft_blur(img_f, maxv, params.get("strength", 0.5), params.get("ksize", 9))
        elif op == "Sobel Edges":
            out_f = effect_edge_sobel(img_f, maxv, params.get("threshold", 0.35))
        elif op == "Pixelate":
            out_f = effect_pixelate(img_f, maxv, params.get("block", 10))
        elif op == "Hue Shift":
            out_f = effect_hue_shift(img_f, maxv, params.get("shift", 0.2))
        else:
            out_f = img_f

        return _clamp_like(image_data, out_f)
ImageModule = ZeiadAbdelkarimImageModule
