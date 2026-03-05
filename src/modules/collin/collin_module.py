from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QPushButton, QComboBox, QStackedWidget, QSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule

# --- Gaussian Blur ---
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def convolve2d_numpy(image_channel: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    img_h, img_w = image_channel.shape
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    padded = np.pad(image_channel, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    output = np.zeros_like(image_channel, dtype=float)
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    return output

def apply_gaussian_blur_numpy(image_data: np.ndarray, sigma: float) -> np.ndarray:
    kernel_size = max(3, int(6 * sigma + 1) | 1)
    kernel = gaussian_kernel(kernel_size, sigma)
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        processed[:, :, c] = convolve2d_numpy(image_data[:, :, c].astype(float), kernel)
    return processed.astype(image_data.dtype)

# --- Box Blur ---
def apply_box_blur(image_data: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=float) / (kernel_size * kernel_size)
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        processed[:, :, c] = convolve2d_numpy(image_data[:, :, c].astype(float), kernel)
    return processed.astype(image_data.dtype)

# --- Sharpening ---
def apply_sharpening(image_data: np.ndarray, strength: float) -> np.ndarray:
    kernel = np.array([[0, -1,  0],
                       [-1,  4, -1],
                       [0, -1,  0]], dtype=float)
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        channel = image_data[:, :, c].astype(float)
        edges = convolve2d_numpy(channel, kernel)
        sharpened = channel + strength * edges
        processed[:, :, c] = np.clip(sharpened, 0, 255)
    return processed.astype(image_data.dtype)

# --- Ideal Low Pass Filter ---
def apply_ideal_low_pass(image_data: np.ndarray, cutoff: float) -> np.ndarray:
    processed = np.zeros_like(image_data, dtype=float)
    rows, cols = image_data.shape[:2]
    u = np.fft.fftfreq(rows) * rows
    v = np.fft.fftfreq(cols) * cols
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    mask = (D <= cutoff).astype(float)
    for c in range(image_data.shape[2]):
        channel = image_data[:, :, c].astype(float)
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        filtered = f_shifted * np.fft.fftshift(mask)
        restored = np.fft.ifft2(np.fft.ifftshift(filtered))
        processed[:, :, c] = np.clip(np.abs(restored), 0, 255)
    return processed.astype(image_data.dtype)

# --- Laplacian Filter ---
def apply_laplacian(image_data: np.ndarray) -> np.ndarray:
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=float)
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        channel = image_data[:, :, c].astype(float)
        edges = convolve2d_numpy(channel, kernel)
        edges = np.abs(edges)
        if edges.max() > 0:
            edges = (edges / edges.max()) * 255
        processed[:, :, c] = edges
    return processed.astype(image_data.dtype)

# --- Histogram Equalisation ---
def apply_histogram_equalisation(image_data: np.ndarray) -> np.ndarray:
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        channel = image_data[:, :, c].astype(float)
        # Compute histogram with 256 bins
        hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 255))
        # Compute cumulative distribution function (CDF)
        cdf = hist.cumsum()
        # Normalize CDF to range [0, 255]
        cdf_min = cdf[cdf > 0].min()
        total_pixels = channel.size
        cdf_normalized = (cdf - cdf_min) / (total_pixels - cdf_min) * 255
        # Map original pixel values to equalised values
        processed[:, :, c] = cdf_normalized[channel.astype(int).clip(0, 255)]
    return np.clip(processed, 0, 255).astype(image_data.dtype)

# --- Laplacian Sharpening ---
def apply_laplacian_sharpening(image_data: np.ndarray, strength: float) -> np.ndarray:
    # Full Laplacian kernel (includes diagonals)
    kernel = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]], dtype=float)
    processed = np.zeros_like(image_data, dtype=float)
    for c in range(image_data.shape[2]):
        channel = image_data[:, :, c].astype(float)
        laplacian = convolve2d_numpy(channel, kernel)
        # Add Laplacian edges back to original image scaled by strength
        sharpened = channel + strength * laplacian
        processed[:, :, c] = np.clip(sharpened, 0, 255)
    return processed.astype(image_data.dtype)

# --- Parameter Widgets ---
class NoParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class GaussianParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setMinimum(0.1)
        self.sigma_spinbox.setMaximum(25.0)
        self.sigma_spinbox.setValue(1.0)
        self.sigma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'sigma': self.sigma_spinbox.value()}

class BoxBlurParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Kernel Size (odd number):"))
        self.size_spinbox = QSpinBox()
        self.size_spinbox.setMinimum(3)
        self.size_spinbox.setMaximum(31)
        self.size_spinbox.setValue(3)
        self.size_spinbox.setSingleStep(2)
        layout.addWidget(self.size_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'kernel_size': self.size_spinbox.value()}

class IdealLowPassParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Cutoff Frequency (pixels):"))
        self.cutoff_spinbox = QDoubleSpinBox()
        self.cutoff_spinbox.setMinimum(1.0)
        self.cutoff_spinbox.setMaximum(500.0)
        self.cutoff_spinbox.setValue(50.0)
        self.cutoff_spinbox.setSingleStep(5.0)
        layout.addWidget(self.cutoff_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'cutoff': self.cutoff_spinbox.value()}

class LaplacianSharpeningParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Strength:"))
        self.strength_spinbox = QDoubleSpinBox()
        self.strength_spinbox.setMinimum(0.1)
        self.strength_spinbox.setMaximum(5.0)
        self.strength_spinbox.setValue(1.0)
        self.strength_spinbox.setSingleStep(0.1)
        layout.addWidget(self.strength_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'strength': self.strength_spinbox.value()}

# --- Control Widget ---
class CollinControlsWidget(QWidget):
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

        operations = {
            "Gaussian Blur":            GaussianParamsWidget,
            "Box Blur":                 BoxBlurParamsWidget,
            "Ideal Low Pass":           IdealLowPassParamsWidget,
            "Laplacian Filter":         NoParamsWidget,
            "Histogram Equalisation":   NoParamsWidget,
            "Laplacian Sharpening":     LaplacianSharpeningParamsWidget,
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_button)

        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)

    def _on_apply_clicked(self):
        operation_name = self.operation_selector.currentText()
        params = self.param_widgets[operation_name].get_params()
        params['operation'] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

# --- Main Module ---
class CollinImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Collin Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = CollinControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get('operation')

        if operation == "Gaussian Blur":
            return apply_gaussian_blur_numpy(image_data, sigma=params.get('sigma', 1.0))
        elif operation == "Box Blur":
            return apply_box_blur(image_data, kernel_size=params.get('kernel_size', 3))
        elif operation == "Ideal Low Pass":
            return apply_ideal_low_pass(image_data, cutoff=params.get('cutoff', 50.0))
        elif operation == "Laplacian Filter":
            return apply_laplacian(image_data)
        elif operation == "Histogram Equalisation":
            return apply_histogram_equalisation(image_data)
        elif operation == "Laplacian Sharpening":
            return apply_laplacian_sharpening(image_data, strength=params.get('strength', 1.0))

        return image_data