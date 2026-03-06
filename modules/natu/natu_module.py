from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# --- Parameter Widgets for Different Operations ---
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

class GaussianParamsWidget(BaseParamsWidget):
    """A widget for Gaussian blur parameters."""
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

class PowerLawParamsWidget(BaseParamsWidget):
    """A widget for Power Law (Gamma) Transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Gamma:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setMinimum(0.01)
        self.gamma_spinbox.setMaximum(5.0)
        self.gamma_spinbox.setValue(1.0)
        self.gamma_spinbox.setSingleStep(0.1)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}

class ConvolutionParamsWidget(BaseParamsWidget):
    """A widget for defining a 3x3 convolution kernel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("3x3 Kernel:"))
        
        grid_layout = QGridLayout()
        self.kernel_inputs = []
        for r in range(3):
            row_inputs = []
            for c in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setMinimum(-100.0)
                spinbox.setMaximum(100.0)
                spinbox.setValue(0.0)
                # Set center to 1.0 for an identity-like default
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}

class MedianParamsWidget(BaseParamsWidget):
    """A widget for median filter radius."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.value_label = QLabel("Radius: 2")
        layout.addWidget(self.value_label)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setValue(2)
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"Radius: {v}"))
        layout.addWidget(self.slider)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'radius': self.slider.value()}

class PosterizeParamsWidget(BaseParamsWidget):
    """A widget for posterize parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.value_label = QLabel("Levels: 4")
        layout.addWidget(self.value_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(2, 16)
        self.slider.setValue(4)
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"Levels: {v}"))
        layout.addWidget(self.slider)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'levels': self.slider.value()}

class SolarizeParamsWidget(BaseParamsWidget):
    """A widget for solarize threshold."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Threshold (0-255):"))
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 255.0)
        self.threshold.setValue(128.0)
        self.threshold.setSingleStep(1.0)
        layout.addWidget(self.threshold)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'threshold': self.threshold.value()}

class ChromaticAbberationParamsWidget(BaseParamsWidget):
    """A widget for chromatic abberation shifts."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Horizontal Shift (px):"))
        self.shift = QSpinBox()
        self.shift.setRange(0, 50)
        self.shift.setValue(5)
        layout.addWidget(self.shift)
        layout.addWidget(QLabel("Vertical Shift (px):"))
        self.vshift = QSpinBox()
        self.vshift.setRange(0, 50)
        self.vshift.setValue(0)
        layout.addWidget(self.vshift)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'shift': self.shift.value(), 'vshift': self.vshift.value()}

class ContrastParamsWidget(BaseParamsWidget):
    """A widget for contrast strength."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Contrast Factor:"))
        self.factor = QDoubleSpinBox()
        self.factor.setRange(0.1, 4.0)
        self.factor.setValue(1.2)
        self.factor.setSingleStep(0.1)
        layout.addWidget(self.factor)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'factor': self.factor.value()}

class ChannelSwapParamsWidget(BaseParamsWidget):
    """A widget for selecting channel swap mode."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Swap Mode:"))
        self.mode = QComboBox()
        self.mode.addItems(["RGB->BGR", "RGB->GRB", "RGB->BRG", "RGB->RBG"])
        layout.addWidget(self.mode)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'mode': self.mode.currentText()}

# Define a custom control widget
class NatuControlsWidget(QWidget):
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
            "Gaussian Blur": GaussianParamsWidget,
            "Median Filter": MedianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Infrared": NoParamsWidget,
            "Posterize": PosterizeParamsWidget,
            "Solarize": SolarizeParamsWidget,
            "Chromatic Abberation": ChromaticAbberationParamsWidget,
            "Contrast": ContrastParamsWidget,
            "Channel Swaps": ChannelSwapParamsWidget,
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

class NatuImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Natu Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = NatuControlsWidget(module_manager, parent)
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

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            # skimage.filters.gaussian expects float data
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        elif operation == "Median Filter":
            radius = int(params.get('radius', 2))
            footprint = skimage.morphology.disk(radius)
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                channels = []
                for i in range(processed_data.shape[2]):
                    channels.append(skimage.filters.median(processed_data[:, :, i], footprint=footprint))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=footprint)
        elif operation == "Sobel Edge Detect":
            # Sobel works on 2D (grayscale) images. Convert if necessary.
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data
            
            processed_data = skimage.filters.sobel(grayscale_img)
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            # Normalize to [0, 1]
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                normalized = input_float / max_val
                # Apply gamma correction
                gamma_corrected = np.power(normalized, gamma)
                # Scale back to original range
                processed_data = gamma_corrected * max_val
        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                # Convolve works best on float images
                input_float = processed_data.astype(float)
                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]: # RGB/RGBA
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')
        elif operation == "Infrared":
            processed_data = self._apply_infrared_numpy(processed_data, params)
        elif operation == "Posterize":
            levels = max(2, int(params.get('levels', 4)))
            max_val = 255.0
            if np.issubdtype(processed_data.dtype, np.floating) and np.max(processed_data) <= 1.0:
                max_val = 1.0
            step = max_val / (levels - 1)
            processed_data = np.round(processed_data.astype(float) / step) * step
            processed_data = np.clip(processed_data, 0.0, max_val)
        elif operation == "Solarize":
            max_val = 255.0
            if np.issubdtype(processed_data.dtype, np.floating) and np.max(processed_data) <= 1.0:
                max_val = 1.0
            threshold = float(params.get('threshold', max_val / 2.0))
            src = processed_data.astype(float)
            processed_data = np.where(src > threshold, max_val - src, src)
            processed_data = np.clip(processed_data, 0.0, max_val)
        elif operation == "Chromatic Abberation":
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                shift = int(params.get('shift', 5))
                vshift = int(params.get('vshift', 0))
                out = processed_data.copy()
                out[:, :, 0] = np.roll(processed_data[:, :, 0], (vshift, shift), axis=(0, 1))
                out[:, :, 2] = np.roll(processed_data[:, :, 2], (-vshift, -shift), axis=(0, 1))
                processed_data = out
        elif operation == "Contrast":
            factor = float(params.get('factor', 1.2))
            src = processed_data.astype(float)
            mean_val = np.mean(src, axis=(0, 1), keepdims=True) if src.ndim == 3 else np.mean(src)
            processed_data = mean_val + factor * (src - mean_val)
            max_val = 255.0
            if np.issubdtype(processed_data.dtype, np.floating) and np.max(image_data) <= 1.0:
                max_val = 1.0
            processed_data = np.clip(processed_data, 0.0, max_val)
        elif operation == "Channel Swaps":
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                mode = params.get('mode', 'RGB->BGR')
                swap_map = {
                    'RGB->BGR': [2, 1, 0],
                    'RGB->GRB': [1, 0, 2],
                    'RGB->BRG': [2, 0, 1],
                    'RGB->RBG': [0, 2, 1],
                }
                order = swap_map.get(mode, [2, 1, 0])
                swapped = processed_data[:, :, order]
                if processed_data.shape[2] > 3:
                    processed_data = np.dstack((swapped, processed_data[:, :, 3:]))
                else:
                    processed_data = swapped


        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data

    def _apply_infrared_numpy(self, data: np.ndarray, params: dict = None) -> np.ndarray:
        input_float = data.astype(np.float32)
        alpha_channel = None

        if input_float.ndim == 3 and 1 in input_float.shape:
            input_float = np.squeeze(input_float)

        if input_float.ndim == 2:
            intensity = input_float
        elif input_float.ndim == 3:
            if input_float.shape[2] in [3, 4]:
                rgb = input_float[:, :, :3]
                if input_float.shape[2] == 4:
                    alpha_channel = input_float[:, :, 3]
                intensity = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            elif input_float.shape[0] in [3, 4]:
                rgb = np.transpose(input_float[:3], (1, 2, 0))
                if input_float.shape[0] == 4:
                    alpha_channel = input_float[3]
                intensity = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
            else:
                intensity = np.mean(input_float, axis=-1)
        else:
            return data

        min_v, max_v = np.min(intensity), np.max(intensity)
        norm = np.clip((intensity - min_v) / max(max_v - min_v, 1e-6), 0.0, 1.0)

        x        = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        r_points = np.array([0.0, 0.6,  1.0, 1.0,  1.0], dtype=np.float32)
        g_points = np.array([0.0, 0.0,  0.4, 1.0,  1.0], dtype=np.float32)
        b_points = np.array([0.0, 0.0,  0.0, 0.0,  1.0], dtype=np.float32)

        infrared = np.stack([
            np.interp(norm, x, r_points),
            np.interp(norm, x, g_points),
            np.interp(norm, x, b_points),
        ], axis=-1)

        if alpha_channel is not None:
            if np.issubdtype(data.dtype, np.integer):
                alpha_norm = np.clip(alpha_channel / float(np.iinfo(data.dtype).max), 0.0, 1.0)
            else:
                alpha_norm = np.clip(alpha_channel, 0.0, 1.0)
            infrared = np.dstack([infrared, alpha_norm])

        if np.issubdtype(data.dtype, np.integer):
            max_out = float(np.iinfo(data.dtype).max)
            infrared = np.clip(infrared * max_out, 0, max_out)

        return infrared
