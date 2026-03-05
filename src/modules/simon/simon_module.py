from PySide6.QtWidgets import QLineEdit, QSpinBox, QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage import filters, color, exposure, morphology
import skimage
from numpy.lib.stride_tricks import sliding_window_view
from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore
# --- Parameter Widgets for Different Operations ---

def layer_mapper(image, thresholds=[64, 128, 192], kernel_size=3, sigma=1.0):
    if image.ndim == 3 :
        grayscale_img = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
    else:
        grayscale_img = image

    if kernel_size % 2 == 0:
        kernel_size += 1

    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    gauss = np.exp(-0.5 * (ax / sigma)**2)

    kernel_2d = np.outer(gauss, gauss)
    kernel_2d /= kernel_2d.sum() 

    pad = kernel_size // 2
    padded_img = np.pad(grayscale_img, pad, mode='edge')

    output = np.zeros_like(image, dtype=float)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extracts the square neighborhood based on kernel_size
            region = padded_img[i:i+kernel_size, j:j+kernel_size]
            output[i, j] = np.sum(region * kernel_2d)
            
    blurred_img = output.astype(np.uint8)

    n_layers = len(thresholds) + 1
    material_map = np.zeros_like(blurred_img)

    thresholds = sorted(thresholds)

    for i, t in enumerate(thresholds):
        # Everything greater than the current threshold gets assigned a higher ID
        material_map[blurred_img > t] = i + 1

    final_gray = (material_map * (255.0 / (n_layers - 1))).astype(np.uint8)

    return np.clip(final_gray, 0, 255).astype(np.uint8)


def apply_thermal_filter(image_data, contrast_factor=0.8):
    # 1. Grayscale Conversion (Manual RGB to Gray)
    if image_data.ndim == 3:
        # Standard ITU-R 601-2 luma transform
        gray = (0.299 * image_data[:,:,0] + 
                0.587 * image_data[:,:,1] + 
                0.114 * image_data[:,:,2]) / 255.0
    else:
        gray = image_data.astype(float) / 255.0

    # 2. Robust Scaling (Percentile Clipping)
    p2, p98 = np.percentile(gray, (2, 98))
    
    if p98 <= p2:
        enhanced = np.zeros_like(gray)
    else:
        # Normalize and clip to [0, 1]
        enhanced = np.clip((gray - p2) / (p98 - p2), 0, 1)

    # 3. Thermal Color Mapping
    thresholds = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    colors = np.array([
        [0, 0, 0],       # Black (Cold)
        [31, 12, 72],    # Dark Purple
        [177, 42, 68],   # Magenta/Red
        [253, 164, 40],  # Orange/Yellow
        [252, 255, 164]  # Pale Yellow (Hot)
    ])

    # Initialize the RGB output array
    h, w = enhanced.shape
    thermal_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Interpolate each channel separately
    for i in range(3): # Red, Green, Blue
        thermal_img[:, :, i] = np.interp(enhanced, thresholds, colors[:, i]).astype(np.uint8)

    return thermal_img



def apply_color_isolation(image_data, target_hue=0.0, tolerance=0.1):
    # 1. Normalize image
    img_float = image_data.astype(float) / 255.0
    
    # 2. Extract Channels
    r, g, b = img_float[:,:,0], img_float[:,:,1], img_float[:,:,2]
    max_c = np.max(img_float, axis=2)
    min_c = np.min(img_float, axis=2)
    delta = max_c - min_c

    # 3. Calculate Hue (Pure NumPy)
    # We use np.zeros and update where delta > 0 to avoid division by zero
    hue = np.zeros_like(max_c)
    
    mask_r = (max_c == r) & (delta != 0)
    mask_g = (max_c == g) & (delta != 0)
    mask_b = (max_c == b) & (delta != 0)

    hue[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
    hue[mask_g] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2
    hue[mask_b] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4
    
    hue = hue / 6.0  # Normalize Hue to [0, 1]

    # 4. Create the Color Mask
    # Handle the circular nature of Hue (e.g., Red is at both 0.0 and 1.0)
    hue_diff = np.abs(hue - target_hue)
    hue_diff = np.minimum(hue_diff, 1.0 - hue_diff)
    color_mask = hue_diff < tolerance

    # 5. Create Grayscale Background
    # Standard Luma conversion: 0.299R + 0.587G + 0.114B
    gray_val = 0.299 * r + 0.587 * g + 0.114 * b
    # Stack to create a 3-channel grayscale image
    gray_img = np.stack([gray_val] * 3, axis=-1)

    # 6. Combine: Isolated Color + Grayscale Background
    result = gray_img.copy()
    # Apply the mask: where the mask is True, use the original color
    result[color_mask] = img_float[color_mask]
    
    return (result * 255).astype(np.uint8)


def apply_sharpening_filter(image_data, weight=1.5):
    img_float = image_data.astype(float) / 255.0
    
    # 1. RGB to XYZ Space
    # We use the standard D65 illuminant matrix
    matrix = np.array([
        [0.412453, 0.357580, 0.180423],
        [0.212671, 0.715160, 0.072169],
        [0.019334, 0.119193, 0.950227]
    ])
    # Reshape for matrix multiplication: (H*W, 3) @ (3, 3)
    shape = img_float.shape
    xyz = (img_float.reshape(-1, 3) @ matrix.T).reshape(shape)

    # 2. XYZ to LAB (L-channel only)
    # Normalized for D65 white point [95.047, 100.0, 108.883]
    y_norm = xyz[:, :, 1] / 1.0  # Y is already normalized 0-1 here
    
    def f(t):
        # Non-linear scaling function for CIE Lab
        mask = t > 0.008856
        res = np.zeros_like(t)
        res[mask] = np.power(t[mask], 1/3)
        res[~mask] = (7.787 * t[~mask]) + (16 / 116)
        return res

    l_channel = (116 * f(y_norm)) - 16

    # 3. Laplace Filter (Discrete Convolution)
    # A standard 3x3 Laplacian kernel for edge detection
    kernel = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]])
    
    # Manual 2D Convolution using NumPy's sliding_window_view
    pad_l = np.pad(l_channel, 1, mode='edge')
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(pad_l, (3, 3))
    edges = np.sum(windows * kernel, axis=(2, 3))

    # 4. Apply Sharpening to L-channel
    sharpened_l = l_channel - (weight * edges)

    # 5. Backtrack to RGB (Simplified for performance)
    # Since we only modified L, we can approximate the change in RGB 
    # by scaling based on the ratio of sharpened_L / original_L
    ratio = np.divide(sharpened_l, l_channel, out=np.ones_like(l_channel), where=l_channel!=0)
    result_rgb = np.clip(img_float * ratio[:, :, np.newaxis], 0, 1)

    return (result_rgb * 255).astype(np.uint8)

def apply_image_negative(image_data):

    negative = 255 - image_data
    return negative.astype(np.uint8)
    


def apply_butterworth_highpass(image_data, d0 = 30, n=2):
    
    img_float = image_data.astype(np.float64)
    rows, cols, channels = img_float.shape
    crow, ccol = rows // 2, cols // 2
    u, v = np.meshgrid(np.arange(cols) - ccol, np.arange(rows) - crow)
    dist = np.sqrt(u**2 + v**2)
    dist[dist == 0] = 1e-10
    mask = 1 / (1 + (d0 / dist)**(2 * n))
    output = np.zeros_like(img_float)

    for i in range(3):
        f_coeff = np.fft.fft2(img_float[:, :, i])
        f_shift = np.fft.fftshift(f_coeff)
        f_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_filtered)
        img_back = np.fft.ifft2(f_ishift)
        output[:, :, i] = np.abs(img_back)

    return np.clip(output, 0, 255).astype(np.uint8)


def apply_hessian_trace(image_data):
    img_float = image_data.astype(np.float64)
    output = np.zeros_like(img_float)

    for i in range(3):
        channel = img_float[..., i]
        dy, dx = np.gradient(channel)
        dyy, _ = np.gradient(dy) # Second derivative in y
        _, dxx = np.gradient(dx) # Second derivative in x
        trace = dxx + dyy
        t_min, t_max = trace.min(), trace.max()
        if t_max > t_min:
            output[..., i] = 255 * (trace - t_min) / (t_max - t_min)
        else:
            output[..., i] = 0

    return output.astype(np.uint8)

def apply_histogram_equisation(image_data):
    r, g, b = image_data[..., 0], image_data[..., 1], image_data[..., 2]
    luma = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    hist, _ = np.histogram(luma.flatten(), 256, [0, 256])
    cumul = hist.cumsum()
    total_pixels = luma.size
    cumul_m = np.ma.masked_equal(cumul, 0)
    lut = ((cumul_m - cumul_m.min()) * 255 / (cumul_m.max() - cumul_m.min()))
    lut = np.ma.filled(lut, 0).astype('uint8')
    equalized_luma = lut[luma]
    old_luma = luma.astype(float)
    old_luma[old_luma == 0] = 1e-10
    ratio = equalized_luma.astype(float) / old_luma
    result = image_data.astype(float) * ratio[..., np.newaxis]
    return np.clip(result, 0, 255).astype(np.uint8)

class BaseParamsWidget(QWidget):
    """Base class for parameter widgets to ensure a consistent interface."""
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    """A placeholder widget for operations with no parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}
    
class DiscreteLayerMapperParamsWidget(BaseParamsWidget):
    """A widget for defining thresholds, blur kernel, and sigma for inlay mapping."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. Layer Thresholds Input
        layout.addWidget(QLabel("Layer Thresholds (comma-separated):"))
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("e.g., 50, 150, 200")
        layout.addWidget(self.threshold_input)

        # 2. Kernel Size Input (Must be Odd)
        layout.addWidget(QLabel("Blur Kernel Size (Odd number):"))
        self.kernel_spin = QSpinBox()
        self.kernel_spin.setRange(1, 31)
        self.kernel_spin.setSingleStep(2) # Ensures the user picks odd numbers
        self.kernel_spin.setValue(5)
        layout.addWidget(self.kernel_spin)

        # 3. Sigma Input (Blur Intensity)
        layout.addWidget(QLabel("Blur Sigma (Intensity):"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 10.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)
        layout.addWidget(self.sigma_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        thresholds_str = self.threshold_input.text()
        try:
            # Parse the comma-separated string into a list of floats
            thresholds = [float(t.strip()) for t in thresholds_str.split(',') if t.strip()]
        except ValueError:
            thresholds = []

        return {
            'thresholds': thresholds,
            'kernel_size': self.kernel_spin.value(),
            'sigma': self.sigma_spin.value()
        }


class IntensitytoColorMappingParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Contrast Factor:"))
        self.contrastFactor_spin = QDoubleSpinBox()
        self.contrastFactor_spin.setRange(0.1, 10.0)
        self.contrastFactor_spin.setSingleStep(0.1)
        self.contrastFactor_spin.setValue(0.8)
        layout.addWidget(self.contrastFactor_spin)
        layout.addStretch()

    
    def get_params(self) -> dict:
        return {'contrastFactor': self.contrastFactor_spin.value()}

class ApplyColorIsolationParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Select color filter:"))

        self.colorDropdown = QComboBox()

        self.colorDropdown.addItem("Red",0)
        self.colorDropdown.addItem("Green",0.33)
        self.colorDropdown.addItem("Blue",0.66)
        layout.addWidget(self.colorDropdown)

        layout.addStretch()

    
    def get_params(self) -> dict:
        return {'color_filter': self.colorDropdown.currentData()}

class ApplySharpeningFilterParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Sharpening Weight:"))
        self.weight_spin = QDoubleSpinBox()
        self.weight_spin.setRange(0.1, 10.0)
        self.weight_spin.setSingleStep(0.1)
        self.weight_spin.setValue(1.0)
        layout.addWidget(self.weight_spin)
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {'weight': self.weight_spin.value()}
    
class ImageNegativeParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("No Parameters are Required"))
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {}
    
class ButterworthHighpassParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Cutoff Frequency:"))
        self.cutoffFrequency_spin = QDoubleSpinBox()
        self.cutoffFrequency_spin.setRange(10, 300)
        self.cutoffFrequency_spin.setSingleStep(1)
        self.cutoffFrequency_spin.setValue(30)
        layout.addWidget(self.cutoffFrequency_spin)

        layout.addWidget(QLabel("Order:"))
        self.order_spin = QDoubleSpinBox()
        self.order_spin.setRange(1, 10)
        self.order_spin.setSingleStep(1)
        self.order_spin.setValue(2)
        layout.addWidget(self.order_spin)
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {
            'cutoff': self.cutoffFrequency_spin.value(),
            'order' : self.order_spin.value()
        }
    
class HessianTraceParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("No Parameters are Required"))
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {}
    
class RGBHistogramEqualisationParamsWidget(BaseParamsWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("No Parameters are Required"))
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {}

# Define a custom control widget
class SimonControlsWidget(QWidget):
    # Signal to request processing from the module manager
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

        # Stacked widget to hold the parameter UIs
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define operations and their corresponding parameter widgets
        operations = {
            "Discrete Layer Mapping": DiscreteLayerMapperParamsWidget,
            "Intensity to Color Mapping": IntensitytoColorMappingParamsWidget,
            "Apply Color Isolation": ApplyColorIsolationParamsWidget,
            "Apply Sharpening Filter": ApplySharpeningFilterParamsWidget,
            "Image Negative": ImageNegativeParamsWidget,
            "Butterworth Highpass": ButterworthHighpassParamsWidget,
            "Hessian Trace": HessianTraceParamsWidget,
            "RGB Histogram Equalisation": RGBHistogramEqualisationParamsWidget,
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
        active_widget = self.param_widgets[operation_name]
        params = active_widget.get_params()
        params['operation'] = operation_name # Add operation name to params
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

class SimonImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Simon Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = SimonControlsWidget(module_manager, parent)
            # The widget's signal is connected to the module's handler
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        # Here, the module needs a way to trigger processing in the main app
        # The control widget now has a valid reference to the module manager
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            # Ensure 2D images are correctly shaped (e.g., handle grayscale vs RGB)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]: # RGB or RGBA
                # napari handles this well, but for processing, sometimes a single channel is needed
                pass
            elif image_data.ndim == 2: # Grayscale
                image_data = image_data[np.newaxis, :] # Add a channel dimension for consistency if desired
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            # Add more metadata: original_shape, file_size, etc.
            return True, image_data, metadata, None # Session ID generated by store
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()

        operation = params.get('operation')

        if operation == "Discrete Layer Mapping":
            thresholds = params.get('thresholds', [])
            kernel_size = params.get('kernel_size', 5)
            sigma = params.get('sigma', 1.0)
            processed_data = layer_mapper(processed_data, thresholds=thresholds, kernel_size=kernel_size, sigma=sigma)


        elif operation == "Intensity to Color Mapping":
            processed_data = apply_thermal_filter(processed_data,params.get('contrastFactor',0.8))

        elif operation == "Apply Color Isolation":
            processed_data = apply_color_isolation(processed_data,params.get('color_filter',0))

        elif operation == "Apply Sharpening Filter":
            processed_data = apply_sharpening_filter(processed_data,params.get('weight', 1))

        elif operation == "Image Negative":
            processed_data = apply_image_negative(processed_data)

        elif operation == "Butterworth Highpass":
            cutoff = params.get('cutoff', 30)
            order = params.get('order', 2)
            processed_data = apply_butterworth_highpass(processed_data, cutoff, order)

        elif operation == "Hessian Trace":
            processed_data = apply_hessian_trace(processed_data)

        elif operation == "RGB Histogram Equalisation":
            processed_data = apply_histogram_equisation(processed_data)

        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data
