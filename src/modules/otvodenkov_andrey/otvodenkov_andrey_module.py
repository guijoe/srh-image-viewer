from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, \
    QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule


# --- NumPy-only Helper Functions ---

def _gaussian_kernel(sigma: float) -> np.ndarray:
    """Build a 2D Gaussian ernel using the outer product of two 1D vectors.
    G(x,y) = exp(-(x²+y²) / 2σ²), then normalised so the sum equals 1.
    """
    radius = max(1, int(np.ceil(3 * sigma)))
    k = np.arange(-radius, radius + 1, dtype=np.float64)
    g1d = np.exp(-k ** 2 / (2.0 * sigma ** 2))
    g1d /= g1d.sum()
    return np.outer(g1d, g1d)


def _fft_convolve2d(channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve a 2D channel with a kernel using FFT (NumPy only).
    Theorem: conv(f,g) = IFFT(FFT(f) * FFT(g)).
    Reflect-padding removes wrap-around border artefacts.
    """
    ih, iw = channel.shape
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    padded = np.pad(channel.astype(np.float64), ((ph, ph), (pw, pw)), mode='reflect')
    fh, fw = padded.shape

    k_embed = np.zeros((fh, fw), dtype=np.float64)
    k_embed[:kh, :kw] = kernel
    k_embed = np.roll(np.roll(k_embed, -(kh // 2), axis=0), -(kw // 2), axis=1)

    result = np.real(np.fft.ifft2(np.fft.fft2(padded) * np.fft.fft2(k_embed)))
    return result[ph: ph + ih, pw: pw + iw]


def _per_channel(image: np.ndarray, fn) -> np.ndarray:
    """Apply a 2D function independently to every colour channel."""
    if image.ndim == 2:
        return fn(image)
    return np.stack([fn(image[:, :, c]) for c in range(image.shape[2])], axis=-1)


def _to_float(img: np.ndarray):
    """Normalise image to float64 in [0, 1]."""
    dtype = img.dtype
    scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    return img.astype(np.float64) / scale, dtype, scale


def _from_float(img: np.ndarray, dtype, scale: float) -> np.ndarray:
    """Convert [0, 1] float back to the original dtype."""
    return np.clip(img * scale, 0, scale).astype(dtype)


def _rgb_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert float RGB image to grayscale using luminance weights."""
    if image.ndim == 2:
        return image
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def _sobel_magnitude(gray: np.ndarray) -> np.ndarray:
    """Compute Sobel edge magnitude on a 2D float image."""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = _fft_convolve2d(gray, kx)
    gy = _fft_convolve2d(gray, ky)
    return np.sqrt(gx ** 2 + gy ** 2)


# --- Artistic Filter Algorithms ---

def oil_painting(image: np.ndarray, radius: int = 4) -> np.ndarray:
    """Kuwahara filter — creates an oil painting brush-stroke effect.
    For each pixel the neighbourhood is split into 4 overlapping quadrants.
    The mean of the quadrant with the lowest variance is assigned to the pixel.
    Low-variance = most uniform region → preserves edges, flattens texture.
    """
    fimg, dtype, scale = _to_float(image)

    def _kuwahara(channel):
        h, w = channel.shape
        r = radius
        padded = np.pad(channel, r, mode='reflect')
        size = (r + 1) ** 2
        min_var = np.full((h, w), np.inf)
        best_mean = np.zeros((h, w))

        for (dy, dx) in [(0, 0), (0, r), (r, 0), (r, r)]:
            q_sum = np.zeros((h, w))
            q_sum2 = np.zeros((h, w))
            for ky in range(r + 1):
                for kx in range(r + 1):
                    v = padded[dy + ky: dy + ky + h, dx + kx: dx + kx + w]
                    q_sum += v
                    q_sum2 += v * v
            mean = q_sum / size
            var = q_sum2 / size - mean * mean
            mask = var < min_var
            min_var = np.where(mask, var, min_var)
            best_mean = np.where(mask, mean, best_mean)

        return best_mean

    result = _per_channel(fimg, _kuwahara)
    return _from_float(result, dtype, scale)


def watercolor(image: np.ndarray, blur_sigma: float = 1.5, saturation_boost: float = 1.4) -> np.ndarray:
    """Watercolor effect: edge-preserving smoothing + HSV saturation lift + paper grain.
    Edges are detected via Sobel and used as a mask to protect detail from over-blurring.
    Saturation is boosted in HSV space to avoid hue distortion in skin tones.
    """
    fimg, dtype, scale = _to_float(image)

    # Step 1: Smooth with fewer passes to avoid over-blurring
    smoothed = fimg.copy()
    kernel = _gaussian_kernel(blur_sigma)
    for _ in range(2):
        smoothed = _per_channel(smoothed, lambda ch: _fft_convolve2d(ch, kernel))

    # Step 2: Edge mask — preserve sharp boundaries from the original
    if fimg.ndim == 3:
        gray = _rgb_to_gray(fimg)
    else:
        gray = fimg
    edges = _sobel_magnitude(gray)
    if edges.max() > 0:
        edges = edges / edges.max()
    # Soft edge mask: blend back original detail where edges are strong
    edge_mask = np.clip(edges * 2.0, 0.0, 1.0)
    if fimg.ndim == 3:
        edge_mask = edge_mask[:, :, np.newaxis]
    smoothed = smoothed * (1.0 - edge_mask) + fimg * edge_mask

    # Step 3: Saturation boost in HSV space — avoids hue shift on skin tones
    if smoothed.ndim == 3 and smoothed.shape[2] >= 3:
        rgb = smoothed[:, :, :3]
        hsv = _rgb_to_hsv(rgb)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0.0, 1.0)
        boosted_rgb = _hsv_to_rgb(hsv)
        if smoothed.shape[2] == 4:
            smoothed = np.concatenate([boosted_rgb, smoothed[:, :, 3:4]], axis=-1)
        else:
            smoothed = boosted_rgb

    # Step 4: Subtle paper grain
    rng = np.random.default_rng(42)
    texture = rng.normal(0, 0.008, smoothed.shape)
    result = np.clip(smoothed + texture, 0.0, 1.0)

    return _from_float(result, dtype, scale)


def xray_effect(image: np.ndarray, gamma: float = 0.6, noise: float = 0.03) -> np.ndarray:
    """X-ray simulation in three stages:
    1. Luminance-weighted grayscale conversion — simulates uniform radiation absorption.
    2. Inversion + gamma correction: xray = (1 - gray)^γ
       γ < 1 lifts dark regions (soft tissue), γ > 1 crushes them (high-contrast bones).
    3. Poisson-like Gaussian noise mimics quantum noise of a real X-ray detector.
    Blue tint replicates the Agfa/Kodak medical film aesthetic.
    """
    fimg, dtype, scale = _to_float(image)

    gray = _rgb_to_gray(fimg) if fimg.ndim == 3 else fimg

    inverted = 1.0 - gray
    xray = np.power(np.clip(inverted, 0.0, 1.0), gamma)

    rng = np.random.default_rng(0)
    grain = rng.normal(0.0, noise, xray.shape)
    xray = np.clip(xray + grain, 0.0, 1.0)

    if image.ndim == 3:
        r = np.clip(xray * 0.70, 0.0, 1.0)
        g = np.clip(xray * 0.85, 0.0, 1.0)
        b = xray
        xray = np.stack([r, g, b], axis=-1)

    return _from_float(xray, dtype, scale)


def mosaic(image: np.ndarray, block_size: int = 16) -> np.ndarray:
    """Artistic mosaic (pixelation) — divides the image into square tiles
    and replaces each tile with its average colour.
    Simulates stained-glass or ancient tile mosaic artwork.
    Larger block_size → coarser, more abstract result.
    """
    fimg, dtype, scale = _to_float(image)
    result = fimg.copy()
    h, w = fimg.shape[:2]
    bs = max(1, int(block_size))

    for y in range(0, h, bs):
        for x in range(0, w, bs):
            block = fimg[y: y + bs, x: x + bs]
            mean_col = block.mean(axis=(0, 1), keepdims=True)
            result[y: y + bs, x: x + bs] = mean_col

    return _from_float(result, dtype, scale)


def pencil_sketch(image: np.ndarray, blur_sigma: float = 15.0) -> np.ndarray:
    """Pencil sketch via the 'dodge blend' technique:
    sketch = gray / (1 - gaussian_blur(inverted_gray))
    Dividing by the blurred inverse amplifies fine edges and
    suppresses flat regions, replicating hand-drawn pencil lines.
    Higher blur_sigma → softer, lighter strokes.
    """
    fimg, dtype, scale = _to_float(image)
    gray = _rgb_to_gray(fimg) if fimg.ndim == 3 else fimg

    inverted = 1.0 - gray
    kernel = _gaussian_kernel(blur_sigma)
    blurred_inv = _fft_convolve2d(inverted, kernel)

    denominator = np.clip(1.0 - blurred_inv, 1e-6, 1.0)
    sketch = np.clip(gray / denominator, 0.0, 1.0)

    if image.ndim == 3:
        sketch = np.stack([sketch] * image.shape[2], axis=-1)

    return _from_float(sketch, dtype, scale)


def cartoonify(image: np.ndarray, num_colors: int = 6, edge_strength: float = 0.85) -> np.ndarray:
    """Cartoon effect in three stages:
    1. Smooth with repeated Gaussian passes (approximates bilateral filter)
       to flatten colour regions while roughly preserving edges.
    2. Quantise colours into discrete flat levels (cartoon fill).
    3. Overlay Sobel edges as black outlines for the drawn look.
    """
    fimg, dtype, scale = _to_float(image)

    smoothed = fimg.copy()
    kernel = _gaussian_kernel(2.5)
    for _ in range(3):
        smoothed = _per_channel(smoothed, lambda ch: _fft_convolve2d(ch, kernel))

    nc = max(2, int(num_colors))
    quantised = np.clip(np.round(smoothed * nc) / nc, 0.0, 1.0)

    gray = _rgb_to_gray(fimg) if fimg.ndim == 3 else fimg
    edges = _sobel_magnitude(gray)
    if edges.max() > 0:
        edges = edges / edges.max()
    edge_mask = edges > 0.15

    result = quantised.copy()
    result[edge_mask] = result[edge_mask] * (1.0 - edge_strength)

    return _from_float(result, dtype, scale)


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert float RGB [0,1] image to HSV (NumPy only)."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.where(maxc != 0, delta / maxc, 0.0)

    h = np.zeros_like(r)
    mask_r = (maxc == r) & (delta != 0)
    mask_g = (maxc == g) & (delta != 0)
    mask_b = (maxc == b) & (delta != 0)
    h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
    h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2.0
    h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4.0
    h = h / 6.0

    return np.stack([h, s, v], axis=-1)


def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert float HSV image back to RGB (NumPy only)."""
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    h6 = h * 6.0
    i = np.floor(h6).astype(int) % 6
    f = h6 - np.floor(h6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])

    return np.stack([r, g, b], axis=-1)


def abstraction(image: np.ndarray, hue_shift: float = 0.4,
                frequency: float = 3.0, saturation_boost: float = 1.5,
                distortion: float = 8.0) -> np.ndarray:
    """Psychedelic abstraction: hue warp + sine distortion + colour fringing.
    Stage 1 — Spatial warp: each pixel's coordinates are displaced by
              sinusoidal waves before sampling — creates a liquid/melting look.
              dx(y) = distortion · sin(2π · y/H · frequency)
              dy(x) = distortion · sin(2π · x/W · frequency)
    Stage 2 — Hue warp in HSV: H' = (H + shift + 0.3·sin(H·freq·2π)) % 1.0
    Stage 3 — Saturation crush: pushes neutral greys to vivid colour.
    Stage 4 — Per-channel iridescent sine fringing with 120° phase offsets.
    """
    fimg, dtype, scale = _to_float(image)

    if fimg.ndim < 3 or fimg.shape[2] < 3:
        result = np.sin(fimg * frequency * np.pi) * 0.5 + 0.5
        return _from_float(result, dtype, scale)

    h, w = fimg.shape[:2]
    rgb = fimg[:, :, :3].copy()

    xv, yv = np.meshgrid(np.arange(w), np.arange(h))

    dx = distortion * np.sin(2.0 * np.pi * yv / h * frequency)
    dy = distortion * np.sin(2.0 * np.pi * xv / w * frequency)

    src_x = np.clip((xv + dx).astype(int), 0, w - 1)
    src_y = np.clip((yv + dy).astype(int), 0, h - 1)

    warped = rgb[src_y, src_x]

    hsv = _rgb_to_hsv(warped)
    hh = hsv[:, :, 0]
    hsv[:, :, 0] = (hh + hue_shift + 0.3 * np.sin(hh * frequency * 2.0 * np.pi)) % 1.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + saturation_boost), 0.0, 1.0)
    rgb_out = _hsv_to_rgb(hsv)

    for c, phase in enumerate([0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0]):
        rgb_out[:, :, c] = np.clip(
            rgb_out[:, :, c] + 0.12 * np.sin(rgb_out[:, :, c] * frequency * np.pi + phase),
            0.0, 1.0
        )

    if fimg.shape[2] == 4:
        result = np.concatenate([rgb_out, fimg[:, :, 3:4]], axis=-1)
    else:
        result = rgb_out

    return _from_float(result, dtype, scale)


def kaleidoscope(image: np.ndarray, segments: int = 8,
                 zoom: float = 1.0, rotation: float = 0.0) -> np.ndarray:
    """Kaleidoscope effect via polar-space tile mirroring.
    Algorithm:
      1. Shift origin to image centre, convert Cartesian → Polar (r, θ).
      2. Divide full 360° into `segments` equal sectors of width α = 2π/segments.
      3. Fold every angle into the first sector [0, α] using modulo + reflection:
             θ_mod = θ % α
             θ_fold = α - θ_mod  if  θ_mod > α/2  else  θ_mod
         Alternating sectors are flipped so adjacent edges always match seamlessly.
      4. Apply optional rotation offset and zoom on the radius.
      5. Convert back to Cartesian, sample with NumPy advanced indexing
         (nearest-neighbour, clipped to image bounds).
    Higher `segments` → smaller petals, more symmetric mandala-like result.
    `zoom` < 1.0 pulls the pattern inward (more repeats visible).
    `rotation` (radians) spins the whole kaleidoscope.
    """
    fimg, dtype, scale = _to_float(image)
    h, w = fimg.shape[:2]
    cy, cx = h / 2.0, w / 2.0

    xv, yv = np.meshgrid(np.arange(w) - cx, np.arange(h) - cy)

    r = np.sqrt(xv ** 2 + yv ** 2)
    theta = (np.arctan2(yv, xv) + rotation) % (2.0 * np.pi)

    n = max(2, int(segments))
    alpha = 2.0 * np.pi / n
    theta_mod = theta % alpha
    theta_fold = np.where(theta_mod > alpha / 2.0, alpha - theta_mod, theta_mod)

    r_zoomed = r / max(0.01, zoom)

    src_xi = np.clip((np.cos(theta_fold) * r_zoomed + cx).astype(int), 0, w - 1)
    src_yi = np.clip((np.sin(theta_fold) * r_zoomed + cy).astype(int), 0, h - 1)

    return _from_float(fimg[src_yi, src_xi], dtype, scale)


# --- Parameter Widgets ---

class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""

    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    """Placeholder widget for operations with no parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class OilPaintingParamsWidget(BaseParamsWidget):
    """A widget for Oil Painting (Kuwahara filter) parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Brush Radius (1 – 16):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(1.0)
        self.radius_spinbox.setMaximum(16.0)
        self.radius_spinbox.setValue(4.0)
        self.radius_spinbox.setSingleStep(1.0)
        self.radius_spinbox.setDecimals(0)
        layout.addWidget(self.radius_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'radius': int(self.radius_spinbox.value())}


class WatercolorParamsWidget(BaseParamsWidget):
    """A widget for Watercolor parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Diffusion Sigma (0.5 – 10.0):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.5)
        self.sigma_spinbox.setMaximum(10.0)
        self.sigma_spinbox.setValue(1.5)
        self.sigma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sigma_spinbox)

        layout.addWidget(QLabel("Saturation Boost (0.5 – 6.0):"))
        self.saturation_spinbox = QDoubleSpinBox()
        self.saturation_spinbox.setMinimum(0.5)
        self.saturation_spinbox.setMaximum(6.0)
        self.saturation_spinbox.setValue(1.4)
        self.saturation_spinbox.setSingleStep(0.1)
        layout.addWidget(self.saturation_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'blur_sigma': self.sigma_spinbox.value(),
            'saturation_boost': self.saturation_spinbox.value()
        }


class MosaicParamsWidget(BaseParamsWidget):
    """A widget for Mosaic/Pixelation parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Tile Size (4 – 128):"))
        self.block_spinbox = QDoubleSpinBox()
        self.block_spinbox.setMinimum(4.0)
        self.block_spinbox.setMaximum(128.0)
        self.block_spinbox.setValue(16.0)
        self.block_spinbox.setSingleStep(2.0)
        self.block_spinbox.setDecimals(0)
        layout.addWidget(self.block_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'block_size': int(self.block_spinbox.value())}


class PencilSketchParamsWidget(BaseParamsWidget):
    """A widget for Pencil Sketch parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Stroke Softness / Sigma (5 – 60):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(5.0)
        self.sigma_spinbox.setMaximum(60.0)
        self.sigma_spinbox.setValue(15.0)
        self.sigma_spinbox.setSingleStep(1.0)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'blur_sigma': self.sigma_spinbox.value()}


class CartoonifyParamsWidget(BaseParamsWidget):
    """A widget for Cartoonify parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Number of Colours (2 – 32):"))
        self.colors_spinbox = QDoubleSpinBox()
        self.colors_spinbox.setMinimum(2.0)
        self.colors_spinbox.setMaximum(32.0)
        self.colors_spinbox.setValue(6.0)
        self.colors_spinbox.setSingleStep(1.0)
        self.colors_spinbox.setDecimals(0)
        layout.addWidget(self.colors_spinbox)

        layout.addWidget(QLabel("Edge Strength (0.0 – 1.0):"))
        self.edge_spinbox = QDoubleSpinBox()
        self.edge_spinbox.setMinimum(0.0)
        self.edge_spinbox.setMaximum(1.0)
        self.edge_spinbox.setValue(0.85)
        self.edge_spinbox.setSingleStep(0.05)
        layout.addWidget(self.edge_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'num_colors': int(self.colors_spinbox.value()),
            'edge_strength': self.edge_spinbox.value()
        }


class XRayParamsWidget(BaseParamsWidget):
    """A widget for X-Ray Effect parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Gamma — tissue contrast (0.2 – 4.0):"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.2)
        self.gamma_spinbox.setMaximum(4.0)
        self.gamma_spinbox.setValue(0.6)
        self.gamma_spinbox.setSingleStep(0.05)
        layout.addWidget(self.gamma_spinbox)

        layout.addWidget(QLabel("Detector Noise (0.0 – 0.2):"))
        self.noise_spinbox = QDoubleSpinBox()
        self.noise_spinbox.setMinimum(0.0)
        self.noise_spinbox.setMaximum(0.2)
        self.noise_spinbox.setValue(0.03)
        self.noise_spinbox.setSingleStep(0.005)
        self.noise_spinbox.setDecimals(3)
        layout.addWidget(self.noise_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'gamma': self.gamma_spinbox.value(),
            'noise': self.noise_spinbox.value()
        }


class AbstractionParamsWidget(BaseParamsWidget):
    """A widget for Psychedelic Abstraction parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Hue Shift (0.0 – 1.0):"))
        self.hue_spinbox = QDoubleSpinBox()
        self.hue_spinbox.setMinimum(0.0)
        self.hue_spinbox.setMaximum(1.0)
        self.hue_spinbox.setValue(0.4)
        self.hue_spinbox.setSingleStep(0.05)
        layout.addWidget(self.hue_spinbox)

        layout.addWidget(QLabel("Warp Frequency (1 – 20):"))
        self.freq_spinbox = QDoubleSpinBox()
        self.freq_spinbox.setMinimum(1.0)
        self.freq_spinbox.setMaximum(20.0)
        self.freq_spinbox.setValue(3.0)
        self.freq_spinbox.setSingleStep(0.5)
        layout.addWidget(self.freq_spinbox)

        layout.addWidget(QLabel("Saturation Boost (0.0 – 10.0):"))
        self.sat_spinbox = QDoubleSpinBox()
        self.sat_spinbox.setMinimum(0.0)
        self.sat_spinbox.setMaximum(10.0)
        self.sat_spinbox.setValue(1.5)
        self.sat_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sat_spinbox)

        layout.addWidget(QLabel("Distortion (0 – 80):"))
        self.distortion_spinbox = QDoubleSpinBox()
        self.distortion_spinbox.setMinimum(0.0)
        self.distortion_spinbox.setMaximum(80.0)
        self.distortion_spinbox.setValue(8.0)
        self.distortion_spinbox.setSingleStep(1.0)
        layout.addWidget(self.distortion_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'hue_shift': self.hue_spinbox.value(),
            'frequency': self.freq_spinbox.value(),
            'saturation_boost': self.sat_spinbox.value(),
            'distortion': self.distortion_spinbox.value()
        }


class KaleidoscopeParamsWidget(BaseParamsWidget):
    """A widget for Kaleidoscope parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Segments (2 – 32):"))
        self.segments_spinbox = QDoubleSpinBox()
        self.segments_spinbox.setMinimum(2.0)
        self.segments_spinbox.setMaximum(32.0)
        self.segments_spinbox.setValue(8.0)
        self.segments_spinbox.setSingleStep(1.0)
        self.segments_spinbox.setDecimals(0)
        layout.addWidget(self.segments_spinbox)

        layout.addWidget(QLabel("Zoom (0.1 – 4.0):"))
        self.zoom_spinbox = QDoubleSpinBox()
        self.zoom_spinbox.setMinimum(0.1)
        self.zoom_spinbox.setMaximum(4.0)
        self.zoom_spinbox.setValue(1.0)
        self.zoom_spinbox.setSingleStep(0.1)
        layout.addWidget(self.zoom_spinbox)

        layout.addWidget(QLabel("Rotation (0.0 – 6.28 rad):"))
        self.rotation_spinbox = QDoubleSpinBox()
        self.rotation_spinbox.setMinimum(0.0)
        self.rotation_spinbox.setMaximum(6.28)
        self.rotation_spinbox.setValue(0.0)
        self.rotation_spinbox.setSingleStep(0.1)
        layout.addWidget(self.rotation_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'segments': int(self.segments_spinbox.value()),
            'zoom':     self.zoom_spinbox.value(),
            'rotation': self.rotation_spinbox.value()
        }


# --- Control Widget ---

class OtvodenkovAndreyControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Artistic Filters</h3>"))

        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Oil Painting":  OilPaintingParamsWidget,
            "Watercolor":    WatercolorParamsWidget,
            "Mosaic":        MosaicParamsWidget,
            "Pencil Sketch": PencilSketchParamsWidget,
            "Cartoonify":    CartoonifyParamsWidget,
            "X-Ray":         XRayParamsWidget,
            "Abstraction":   AbstractionParamsWidget,
            "Kaleidoscope":  KaleidoscopeParamsWidget,
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

    def _on_apply_clicked(self):
        operation_name = self.operation_selector.currentText()
        params = self.param_widgets[operation_name].get_params()
        params['operation'] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])


# --- Image Module ---

class OtvodenkovAndreyImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Otvodenkov Andrey – Artistic Filters"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = OtvodenkovAndreyControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "Oil Painting":
            processed_data = oil_painting(processed_data, radius=params.get('radius', 4))

        elif operation == "Watercolor":
            processed_data = watercolor(processed_data,
                                        blur_sigma=params.get('blur_sigma', 1.5),
                                        saturation_boost=params.get('saturation_boost', 1.4))

        elif operation == "Mosaic":
            processed_data = mosaic(processed_data, block_size=params.get('block_size', 16))

        elif operation == "Pencil Sketch":
            processed_data = pencil_sketch(processed_data, blur_sigma=params.get('blur_sigma', 15.0))

        elif operation == "Cartoonify":
            processed_data = cartoonify(processed_data,
                                        num_colors=params.get('num_colors', 6),
                                        edge_strength=params.get('edge_strength', 0.85))

        elif operation == "X-Ray":
            processed_data = xray_effect(processed_data,
                                         gamma=params.get('gamma', 0.6),
                                         noise=params.get('noise', 0.03))

        elif operation == "Abstraction":
            processed_data = abstraction(processed_data,
                                         hue_shift=params.get('hue_shift', 0.4),
                                         frequency=params.get('frequency', 3.0),
                                         saturation_boost=params.get('saturation_boost', 1.5),
                                         distortion=params.get('distortion', 8.0))

        elif operation == "Kaleidoscope":
            processed_data = kaleidoscope(processed_data,
                                          segments=params.get('segments', 8),
                                          zoom=params.get('zoom', 1.0),
                                          rotation=params.get('rotation', 0.0))

        return processed_data.astype(image_data.dtype)
