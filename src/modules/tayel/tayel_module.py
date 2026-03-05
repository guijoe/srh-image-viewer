from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
import imageio
import os
from modules.i_image_module import IImageModule

def _f(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64, copy=False)

def _cast_like(orig: np.ndarray, x: np.ndarray) -> np.ndarray:
    if np.issubdtype(orig.dtype, np.integer):
        info = np.iinfo(orig.dtype)
        return np.clip(x, info.min, info.max).astype(orig.dtype)
    return x.astype(orig.dtype, copy=False)

def _pad_reflect(x2d: np.ndarray, ph: int, pw: int) -> np.ndarray:
    return np.pad(x2d, ((ph, ph), (pw, pw)), mode="reflect")

def _apply_per_channel(img: np.ndarray, fn2d, *args, **kwargs) -> np.ndarray:
    """Run fn2d on grayscale or per-channel RGB/RGBA, then cast back."""
    if img.ndim == 2:
        return _cast_like(img, fn2d(img, *args, **kwargs))
    if img.ndim == 3 and img.shape[2] in (3, 4):
        out = np.zeros_like(_f(img))
        for c in range(img.shape[2]):
            out[..., c] = fn2d(img[..., c], *args, **kwargs)
        return _cast_like(img, out)
    return _cast_like(img, fn2d(np.squeeze(img), *args, **kwargs))

def _convolve2d(img2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img = _f(img2d)
    k = np.flip(_f(kernel), (0, 1)) 
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    pad = _pad_reflect(img, ph, pw)

    H, W = img.shape
    out = np.zeros_like(img)

    for i in range(kh):
        for j in range(kw):
            out += pad[i:i + H, j:j + W] * k[i, j]
    return out

def apply_kernel(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return _apply_per_channel(img, _convolve2d, kernel)

def _gaussian_1d(size: int, sigma: float) -> np.ndarray:
    size = max(3, int(size) | 1)
    r = size // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    g = np.exp(-(x * x) / (2.0 * float(sigma) * float(sigma)))
    return g / (g.sum() + 1e-12)

def _conv1d_reflect(img2d: np.ndarray, k1d: np.ndarray, axis: int) -> np.ndarray:
    img = _f(img2d)
    k = _f(k1d)
    r = k.size // 2

    if axis == 1: 
        pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
        out = np.zeros_like(img)
        H, W = img.shape
        for j in range(k.size):
            out += pad[:, j:j + W] * k[j]
        return out

    pad = np.pad(img, ((r, r), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    H, W = img.shape
    for i in range(k.size):
        out += pad[i:i + H, :] * k[i]
    return out

def _gaussian_blur(img: np.ndarray, sigma: float, size: int) -> np.ndarray:
    g = _gaussian_1d(size, sigma)

    def blur2d(ch2d: np.ndarray) -> np.ndarray:
        tmp = _conv1d_reflect(ch2d, g, axis=1)
        return _conv1d_reflect(tmp, g, axis=0)

    return _apply_per_channel(img, blur2d)

def unsharp_mask(img: np.ndarray, sigma: float, amount: float, ksize: int) -> np.ndarray:
    blur = _gaussian_blur(img, sigma=sigma, size=ksize)
    out = _f(img) + float(amount) * (_f(img) - _f(blur))
    return _cast_like(img, out)

def fft_filter(img: np.ndarray, mode: str, cutoff: float) -> np.ndarray:
    cutoff = float(np.clip(cutoff, 0.01, 0.99))

    def f2(ch2d: np.ndarray) -> np.ndarray:
        x = _f(ch2d)
        H, W = x.shape
        F = np.fft.fftshift(np.fft.fft2(x))
        cy, cx = H // 2, W // 2
        yy, xx = np.ogrid[:H, :W]
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = np.sqrt(cy ** 2 + cx ** 2)
        rcut = cutoff * rmax
        mask = (rr <= rcut) if mode == "Low-pass" else (rr >= rcut)
        y = np.fft.ifft2(np.fft.ifftshift(F * mask))
        return np.real(y)

    return _apply_per_channel(img, f2)

def clahe(img: np.ndarray, clip_limit: float, tile_size: int) -> np.ndarray:
    clip_limit = float(clip_limit)
    tile_size = max(8, int(tile_size))

    def clahe_gray(ch2d: np.ndarray) -> np.ndarray:
        g = _f(ch2d)
        H, W = g.shape
        ts = tile_size
        out = np.zeros_like(g)

        for y in range(0, H, ts):
            for x in range(0, W, ts):
                t = g[y:y + ts, x:x + ts]
                hist, _ = np.histogram(t, bins=256, range=(0, 255))
                limit = clip_limit * t.size / 256.0
                excess = np.maximum(hist - limit, 0).sum()
                hist = np.minimum(hist, limit)
                hist += excess / 256.0
                cdf = np.cumsum(hist)
                cdf = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min() + 1e-6)
                out[y:y + ts, x:x + ts] = cdf[t.astype(np.uint8)]
        return out

    return _apply_per_channel(img, clahe_gray)

def solarize(img: np.ndarray, threshold: float) -> np.ndarray:
    x = _f(img)
    maxv = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else float(np.max(x) or 1.0)
    t = float(np.clip(threshold, 0.0, 1.0)) * maxv
    out = x.copy()
    m = out >= t
    out[m] = maxv - out[m]
    return _cast_like(img, out)

def vignette(img: np.ndarray, strength: float, radius: float) -> np.ndarray:
    x = _f(img)
    H, W = x.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    dy = (yy - cy) / (cy + 1e-9)
    dx = (xx - cx) / (cx + 1e-9)
    r = np.sqrt(dx * dx + dy * dy)

    strength = float(np.clip(strength, 0.0, 1.0))
    radius = float(np.clip(radius, 0.2, 1.5))

    mask = 1.0 - strength * (r / radius) ** 2
    mask = np.clip(mask, 1.0 - strength, 1.0)

    out = x * mask if x.ndim == 2 else x * mask[..., None]
    return _cast_like(img, out)

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError

class _SpinRow(BaseParamsWidget):
    """Generic compact widget for (label -> spinbox)."""
    def __init__(self, rows):
        super().__init__()
        lay = QVBoxLayout(self)
        self.items = {}
        for key, label, cls, mn, mx, val, step in rows:
            lay.addWidget(QLabel(label))
            sb = cls()
            sb.setRange(mn, mx)
            sb.setValue(val)
            if hasattr(sb, "setSingleStep"):
                sb.setSingleStep(step)
            lay.addWidget(sb)
            self.items[key] = sb
        lay.addStretch()

    def get_params(self) -> dict:
        out = {}
        for k, sb in self.items.items():
            out[k] = float(sb.value()) if isinstance(sb, QDoubleSpinBox) else int(sb.value())
        return out

class UnsharpParamsWidget(_SpinRow):
    def __init__(self): super().__init__([
        ("sigma",  "Sigma (blur strength):", QDoubleSpinBox, 0.1, 25.0, 1.2, 0.1),
        ("ksize",  "Kernel size (odd):",     QSpinBox,       3,   31,   5,   2),
        ("amount", "Amount (strength):",     QDoubleSpinBox, 0.0, 5.0,  1.5, 0.1),
    ])

class FFTParamsWidget(BaseParamsWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Mode:"))
        self.mode = QComboBox()
        self.mode.addItems(["Low-pass", "High-pass"])
        lay.addWidget(self.mode)

        lay.addWidget(QLabel("Cutoff (0–1):"))
        self.cut = QDoubleSpinBox()
        self.cut.setRange(0.01, 0.99)
        self.cut.setValue(0.25)
        self.cut.setSingleStep(0.01)
        lay.addWidget(self.cut)
        lay.addStretch()

    def get_params(self) -> dict:
        return {"mode": self.mode.currentText(), "cutoff": float(self.cut.value())}

class CLAHEParamsWidget(_SpinRow):
    def __init__(self): super().__init__([
        ("clip_limit", "Clip Limit (0.1–10):", QDoubleSpinBox, 0.1, 10.0, 2.0, 0.1),
        ("tile_size",  "Tile Size (8–128):",   QSpinBox,       8,   128,  32,  1),
    ])

class SolarizeParamsWidget(_SpinRow):
    def __init__(self): super().__init__([
        ("threshold", "Threshold (0–1):", QDoubleSpinBox, 0.0, 1.0, 0.5, 0.01),
    ])

class VignetteParamsWidget(_SpinRow):
    def __init__(self): super().__init__([
        ("strength", "Strength (0–1):", QDoubleSpinBox, 0.0, 1.0, 0.4, 0.01),
        ("radius",   "Radius (0.2–1.5):", QDoubleSpinBox, 0.2, 1.5, 1.0, 0.05),
    ])

class ConvolutionParamsWidget(BaseParamsWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Custom 3×3 Kernel:"))
        grid = QGridLayout()
        self.sbs = []
        for r in range(3):
            row = []
            for c in range(3):
                sb = QDoubleSpinBox()
                sb.setRange(-20.0, 20.0)
                sb.setSingleStep(0.1)
                sb.setValue(1.0 if (r == 1 and c == 1) else 0.0)
                grid.addWidget(sb, r, c)
                row.append(sb)
            self.sbs.append(row)
        lay.addLayout(grid)
        lay.addStretch()

    def get_params(self) -> dict:
        k = np.array([[sb.value() for sb in row] for row in self.sbs], dtype=np.float64)
        return {"kernel": k}

class TayelControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.mm = module_manager
        self.widgets = {}
        self._ui()

    def _ui(self):
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("<h3>Tayel Module</h3>"))
        lay.addWidget(QLabel("Operation:"))

        self.op = QComboBox()
        lay.addWidget(self.op)

        self.stack = QStackedWidget()
        lay.addWidget(self.stack)

        ops = {
            "Unsharp Mask (Sharpen)": UnsharpParamsWidget,
            "FFT Filter (Low/High Pass)": FFTParamsWidget,
            "CLAHE (Local Contrast Enhance)": CLAHEParamsWidget,
            "Solarization": SolarizeParamsWidget,
            "Vignette": VignetteParamsWidget,
            "Custom 3x3 Matrix": ConvolutionParamsWidget,
        }

        for name, cls in ops.items():
            w = cls()
            self.widgets[name] = w
            self.stack.addWidget(w)
            self.op.addItem(name)

        btn = QPushButton("Apply")
        lay.addWidget(btn)

        self.op.currentTextChanged.connect(lambda n: self.stack.setCurrentWidget(self.widgets[n]))
        btn.clicked.connect(self._apply)

    def _apply(self):
        name = self.op.currentText()
        p = dict(self.widgets[name].get_params())
        p["operation"] = name
        self.process_requested.emit(p)

class TayelImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._ctrl = None

    def get_name(self) -> str:
        return "Tayel Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._ctrl is None:
            self._ctrl = TayelControlsWidget(module_manager, parent)
            self._ctrl.process_requested.connect(
                lambda p: module_manager.apply_processing_to_current_image(p)
            )
        return self._ctrl

    def load_image(self, file_path: str):
        try:
            img = imageio.imread(file_path)
            return True, img, {"name": os.path.basename(file_path)}, None
        except Exception as e:
            print(f"Load error: {e}")
            return False, None, {}, None

    def process_image(self, img: np.ndarray, meta: dict, p: dict) -> np.ndarray:
        try:
            op = p.get("operation", "")

            if op == "Unsharp Mask (Sharpen)":
                return unsharp_mask(img, p["sigma"], p["amount"], p["ksize"])

            if op == "FFT Filter (Low/High Pass)":
                return fft_filter(img, p["mode"], p["cutoff"])

            if op == "CLAHE (Local Contrast Enhance)":
                return clahe(img, p["clip_limit"], p["tile_size"])

            if op == "Solarization":
                return solarize(img, p["threshold"])

            if op == "Vignette":
                return vignette(img, p["strength"], p["radius"])

            if op == "Custom 3x3 Matrix":
                return apply_kernel(img, p["kernel"])

            return img

        except Exception as e:
            print(f"Process error in TayelImageModule: {e}")
            return img