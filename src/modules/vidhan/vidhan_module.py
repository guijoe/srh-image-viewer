from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                               QComboBox, QStackedWidget, QDoubleSpinBox, QSpinBox, 
                               QGridLayout, QProgressBar, QApplication)
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio.v3 as iio
import time 

from modules.i_image_module import IImageModule
from modules.sample.sample_module import BaseParamsWidget                                
# 1. THE UI WIDGETS (Parameter Inputs & UX Enhancements)
class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters needed. Just click Apply!"))
        layout.addStretch()
    def get_params(self) -> dict: return {}

class BrightnessParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Brightness Intensity (Min: -255 | Max: 255):"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(-255.0, 255.0)
        self.val_spin.setValue(30.0)
        layout.addWidget(self.val_spin)
        layout.addStretch()
    def get_params(self) -> dict: return {'value': self.val_spin.value()}

class ThresholdParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Level (Min: 0 | Max: 255):"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(0, 255)
        self.val_spin.setValue(127)
        layout.addWidget(self.val_spin)
        layout.addStretch()
    def get_params(self) -> dict: return {'threshold': self.val_spin.value()}

class PowerLawParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma (Min: 0.01 | Max: 5.0):"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.5)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()
    def get_params(self) -> dict: return {'gamma': self.gamma_spinbox.value()}

class ColorProfileParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select Tone/Channel:"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems([
            "Sepia Tone", 
            "Cyanotype (Blueprint)",
            "Monochrome (Grayscale)", 
            "Color Swap (RGB to BGR)",
            "Night Vision (Green Phosphor)",
            "Autumn Warmth (Golden Hour)",
            "Red Channel Only", 
            "Green Channel Only", 
            "Blue Channel Only"
        ])
        layout.addWidget(self.profile_combo)
        layout.addStretch()
    def get_params(self) -> dict: return {'profile': self.profile_combo.currentText()}

class ConvolutionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Custom 3x3 Kernel (Auto-Normalized):"))
        grid_layout = QGridLayout()
        self.kernel_inputs = []
        for r in range(3):
            row_inputs = []
            for c in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-100.0, 100.0)
                spinbox.setValue(1.0 if r==1 and c==1 else 0.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)
        layout.addStretch()
    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}

class CannyParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Low Threshold (Min: 0 | Max: 255):"))
        self.low_spin = QSpinBox()
        self.low_spin.setRange(0, 255)
        self.low_spin.setValue(50)
        layout.addWidget(self.low_spin)

        layout.addWidget(QLabel("High Threshold (Min: 0 | Max: 255):"))
        self.high_spin = QSpinBox()
        self.high_spin.setRange(0, 255)
        self.high_spin.setValue(120)
        layout.addWidget(self.high_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'low': self.low_spin.value(),
            'high': self.high_spin.value()
        }
# 2. THE CONTROL PANEL (Main UI Hub)
class VidhanControlsWidget(QWidget):
    process_requested = Signal(dict)
    
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        
        self.descriptions = {
            "1. Color Inversion (Negative)": "Inverts pixel colors.",
            "2. Brightness Adjustment": "Evenly lightens or darkens the entire image.",
            "3. Binary Thresholding": "Converts the image to pure black and white based on a threshold limit.",
            "4. Power Law (Gamma)": "Non-linearly adjusts mid-tones to fix washed-out images using a high-speed LUT.",
            "5. Color Profiles (Tones & RGB)": "Apply matrix transformations to shift colors to Sepia, Cyanotype, Grayscale, or isolated channels.",
            "6. Solarization": "A darkroom effect that inverts only the pixels above a certain threshold.",
            "7. Vignette Effect": "Darkens the corners using a mathematically generated radial gradient mask.",
            "8. Box Blur (Smoothing)": "Softens the image using an optimized, memory-efficient vectorized 3x3 convolution.",
            "9. Sharpening (Laplacian)": "Enhances edges using a high-performance vectorized 3x3 convolution.",
            "10. Custom Convolution": "Input your own 3x3 matrix to create custom filters! Features Auto-Normalization.",
            "11. Canny Edge Detection": "Multi-stage edge detector using Gaussian smoothing, Sobel gradients, non-max suppression, and hysteresis."
        }
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        header = QLabel("<h3>Vidhan_Module</h3>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        layout.addWidget(QLabel("Select Filter:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        self.description_label = QLabel()
        self.description_label.setWordWrap(True) 
        self.description_label.setStyleSheet("color: #666666; font-style: italic; margin-top: 5px; margin-bottom: 10px;") 
        layout.addWidget(self.description_label)

        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        self.operations = {
            "1. Color Inversion (Negative)": NoParamsWidget,
            "2. Brightness Adjustment": BrightnessParamsWidget,
            "3. Binary Thresholding": ThresholdParamsWidget,
            "4. Power Law (Gamma)": PowerLawParamsWidget,
            "5. Color Profiles (Tones & RGB)": ColorProfileParamsWidget,
            "6. Solarization": ThresholdParamsWidget,
            "7. Vignette Effect": NoParamsWidget,
            "8. Box Blur (Smoothing)": NoParamsWidget,
            "9. Sharpening (Laplacian)": NoParamsWidget,
            "10. Custom Convolution": ConvolutionParamsWidget,
            "11. Canny Edge Detection": CannyParamsWidget
        }

        for name, widget_class in self.operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        # UX UPGRADE: Adding an indeterminate progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # 0,0 makes it an endless spinning bar
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False) # Hidden by default
        layout.addWidget(self.progress_bar)

        self.apply_button = QPushButton("Apply Processing")
        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)
        
        if self.operation_selector.count() > 0:
            self._on_operation_changed(self.operation_selector.currentText())

    def _on_apply_clicked(self):
        op_name = self.operation_selector.currentText()
        params = self.param_widgets[op_name].get_params()
        params['operation'] = op_name
        
        self.apply_button.setText("Processing... Please wait.")
        self.apply_button.setEnabled(False) # Disable to prevent double-clicks
        self.progress_bar.setVisible(True)
        
        # Force the UI to refresh immediately before the heavy math freezes the thread
        QApplication.processEvents()

        # Emit the signal to start processing
        self.process_requested.emit(params)
        
        # Reset UX after processing finishes
        self.apply_button.setText("Apply Processing")
        self.apply_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def _on_operation_changed(self, op_name: str):
        if op_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[op_name])
            self.description_label.setText(self.descriptions.get(op_name, ""))

# 3. THE ALGORITHMS (Core Image Processing Functions)
def _apply_negative(image): 
    return 255 - image

def _apply_brightness(image, value):
    return np.clip(image.astype(np.int16) + value, 0, 255).astype(np.uint8)

def _apply_threshold(image, threshold):
    gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]) if len(image.shape) == 3 else image
    binary = np.where(gray > threshold, 255, 0)
    return np.stack([binary]*3, axis=-1).astype(np.uint8) if len(image.shape) == 3 else binary.astype(np.uint8)

def _apply_gamma(image, gamma):
    lut = np.clip(np.power(np.arange(256) / 255.0, gamma) * 255.0, 0, 255).astype(np.uint8)
    return lut[image]

def _apply_color_profile(image, profile):
    if len(image.shape) < 3: return image 
    
    if profile == "Sepia Tone":
        matrix = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])
        res = np.dot(image.astype(float)[..., :3], matrix.T)
        return np.clip(res, 0, 255).astype(np.uint8)
        
    elif profile == "Cyanotype (Blueprint)":
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        res = np.zeros_like(image, dtype=float)
        res[..., 0] = gray * 0.1 
        res[..., 1] = gray * 0.4 
        res[..., 2] = gray * 0.9 
        return np.clip(res, 0, 255).astype(np.uint8)

    elif profile == "Monochrome (Grayscale)":
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        return np.stack([gray]*3, axis=-1).astype(np.uint8)
        
    elif profile == "Color Swap (RGB to BGR)":
        return image[..., ::-1].copy()

    elif profile == "Night Vision (Green Phosphor)":
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        res = np.zeros_like(image, dtype=float)
        res[..., 1] = gray * 1.2 
        return np.clip(res, 0, 255).astype(np.uint8)

    elif profile == "Autumn Warmth (Golden Hour)":
        res = image.astype(float).copy()
        res[..., 0] = res[..., 0] * 1.2
        res[..., 2] = res[..., 2] * 0.8
        return np.clip(res, 0, 255).astype(np.uint8)
        
    else:
        res = np.zeros_like(image)
        if profile == "Red Channel Only": res[..., 0] = image[..., 0]
        elif profile == "Green Channel Only": res[..., 1] = image[..., 1]
        elif profile == "Blue Channel Only": res[..., 2] = image[..., 2]
        return res

def _apply_solarize(image, threshold):
    return np.where(image < threshold, image, 255 - image).astype(np.uint8)

def _apply_vignette(image):
    rows, cols = image.shape[:2]
    X, Y = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    radius = np.sqrt(X**2 + Y**2)
    mask = 1.0 - np.clip(radius, 0, 1.0)
    if len(image.shape) == 3: mask = np.stack([mask]*3, axis=-1)
    return np.clip(image.astype(float) * mask, 0, 255).astype(np.uint8)

def _apply_convolution(image, kernel):
    k_sum = np.sum(kernel)
    if k_sum != 0 and k_sum != 1:
        kernel = kernel / k_sum
        
    kernel = np.flip(kernel)

    def convolve_2d(channel, k):
        padded = np.pad(channel, 1, mode='edge').astype(np.float32)
        output = np.zeros_like(channel, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                output += padded[i:i+channel.shape[0], j:j+channel.shape[1]] * k[i, j]
        return output

    if len(image.shape) == 2: res = convolve_2d(image, kernel)
    else: res = np.stack([convolve_2d(image[..., c], kernel) for c in range(3)], axis=-1)
    return np.clip(res, 0, 255).astype(np.uint8)

def _apply_canny(image, low, high):
    # 1. Grayscale
    if len(image.shape) == 3 and image.shape[2] >= 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    else:
        gray = image.astype(float)

    # Helper: Fast Vectorized Convolve for Canny
    def fast_convolve_2d(img_2d, k):
        pad_w = k.shape[0] // 2
        padded = np.pad(img_2d, pad_w, mode='edge').astype(np.float32)
        out = np.zeros_like(img_2d, dtype=np.float32)
        for i in range(k.shape[0]):
            for j in range(k.shape[1]):
                out += padded[i:i+img_2d.shape[0], j:j+img_2d.shape[1]] * k[i, j]
        return out

    # 2. Gaussian blur (5x5)
    gaussian_kernel = np.array([
        [2, 4, 5, 4, 2],
        [4, 9,12, 9, 4],
        [5,12,15,12, 5],
        [4, 9,12, 9, 4],
        [2, 4, 5, 4, 2]
    ], dtype=np.float32)
    gaussian_kernel /= np.sum(gaussian_kernel)
    smoothed = fast_convolve_2d(gray, gaussian_kernel)

    # 3. Sobel gradients
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[ 1, 2, 1], [ 0, 0, 0], [-1,-2,-1]], dtype=np.float32)
    
    Gx = fast_convolve_2d(smoothed, Kx)
    Gy = fast_convolve_2d(smoothed, Ky)

    magnitude = np.hypot(Gx, Gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0
    angle = np.arctan2(Gy, Gx)
    angle_deg = np.rad2deg(angle) % 180

    # 4. Non-maximum suppression
    nms = np.zeros_like(magnitude)
    rows, cols = magnitude.shape
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            q = 255
            r = 255
            ang = angle_deg[i,j]
            
            if (0 <= ang < 22.5) or (157.5 <= ang <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            elif 22.5 <= ang < 67.5:
                q = magnitude[i-1, j+1]
                r = magnitude[i+1, j-1]
            elif 67.5 <= ang < 112.5:
                q = magnitude[i-1, j]
                r = magnitude[i+1, j]
            elif 112.5 <= ang < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]

            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                nms[i,j] = magnitude[i,j]

    # 5. Double threshold
    strong = 255
    weak = 50
    result = np.zeros_like(nms)
    
    strong_i, strong_j = np.where(nms >= high)
    weak_i, weak_j = np.where((nms >= low) & (nms < high))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak

    # 6. Hysteresis
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            if result[i,j] == weak:
                if np.max(result[i-1:i+2, j-1:j+2]) == strong:
                    result[i,j] = strong
                else:
                    result[i,j] = 0

    if len(image.shape) == 3:
        return np.stack([result]*3, axis=-1).astype(np.uint8)
    return result.astype(np.uint8)
# 4. THE MAIN MODULE CLASS (Integration & Profiling)

class VidhanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str: return "Vidhan_Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = VidhanControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = iio.imread(file_path)
            if len(image_data.shape) == 2: image_data = np.stack([image_data] * 3, axis=-1)
            elif len(image_data.shape) == 3 and image_data.shape[2] == 4: image_data = image_data[..., :3] 
            return True, image_data, {'name': file_path}, str(id(image_data))
        except Exception as e:
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        op = params.get('operation')
        
        # Start the Benchmark Timer
        start_time = time.perf_counter()
        
        res = image_data
        
        # Dispatch the correct function
        if "Color Inversion" in op: res = _apply_negative(image_data)
        elif "Brightness" in op: res = _apply_brightness(image_data, params.get('value', 0))
        elif "Thresholding" in op: res = _apply_threshold(image_data, params.get('threshold', 127))
        elif "Power Law" in op: res = _apply_gamma(image_data, params.get('gamma', 1.0))
        elif "Color Profiles" in op: res = _apply_color_profile(image_data, params.get('profile', 'Sepia Tone'))
        elif "Solarization" in op: res = _apply_solarize(image_data, params.get('threshold', 128))
        elif "Vignette" in op: res = _apply_vignette(image_data)
        elif "Box Blur" in op: res = _apply_convolution(image_data, np.ones((3, 3), dtype=float) / 9.0)
        elif "Sharpening" in op: res = _apply_convolution(image_data, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        elif "Custom Convolution" in op: res = _apply_convolution(image_data, params.get('kernel'))
        elif "Canny Edge Detection" in op: res = _apply_canny(image_data, params.get('low', 50), params.get('high', 120))
        # Stop the timer and print the benchmark
        end_time = time.perf_counter()
        print(f"[Vidhan_Module] Executed '{op}' perfectly in {(end_time - start_time):.4f} seconds.")

        return res