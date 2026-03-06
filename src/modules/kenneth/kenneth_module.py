from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
# Note: Ensure this import path is correct for your project structure
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

class GenericSliderParamsWidget(BaseParamsWidget):
    """A reusable widget for any operation that just needs one numerical slider."""
    def __init__(self, param_name, min_val, max_val, default_val, step, parent=None):
        super().__init__(parent)
        self.param_name = param_name
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(f"{param_name.capitalize()}:"))
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setMinimum(min_val)
        self.spinbox.setMaximum(max_val)
        self.spinbox.setValue(default_val)
        self.spinbox.setSingleStep(step)
        layout.addWidget(self.spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {self.param_name: self.spinbox.value()}

# Define a custom control widget
class KennethControlsWidget(QWidget):
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
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Brightness": lambda: GenericSliderParamsWidget("beta", -100.0, 100.0, 0.0, 5.0),
            "Contrast": lambda: GenericSliderParamsWidget("alpha", 0.0, 3.0, 1.0, 0.1),
            "Exposure": lambda: GenericSliderParamsWidget("stops", -3.0, 3.0, 0.0, 0.25),
            "Saturation": lambda: GenericSliderParamsWidget("scale", 0.0, 3.0, 1.0, 0.1),
            "Warmth": lambda: GenericSliderParamsWidget("warmth", -50.0, 50.0, 0.0, 2.0),
            "Vignette": lambda: GenericSliderParamsWidget("intensity", 0.0, 1.0, 0.5, 0.1),
            "Sharpness": lambda: GenericSliderParamsWidget("amount", 0.0, 2.0, 1.0, 0.1),
            "Noise Reduction": lambda: GenericSliderParamsWidget("strength", 1.0, 5.0, 1.0, 1.0),
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

class KennethImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Kenneth Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = KennethControlsWidget(module_manager, parent)
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
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
            
        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size > 1:
                if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                    channels = []
                    for i in range(processed_data.shape[2]):
                        channels.append(skimage.filters.median(processed_data[:,:,i], footprint=skimage.morphology.disk(int(filter_size/2))))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = skimage.filters.median(processed_data, footprint=skimage.morphology.disk(int(filter_size/2)))
                    
        elif operation == "Sobel Edge Detect":
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                grayscale_img = rgb2gray(processed_data[:,:,:3])
            else:
                grayscale_img = processed_data
            processed_data = skimage.filters.sobel(grayscale_img)
            
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                normalized = input_float / max_val
                gamma_corrected = np.power(normalized, gamma)
                processed_data = gamma_corrected * max_val
                
        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                input_float = processed_data.astype(float)
                if input_float.ndim == 3 and input_float.shape[2] in [3, 4]: 
                    channels = []
                    for i in range(input_float.shape[2]):
                        channels.append(convolve(input_float[:,:,i], kernel, mode='reflect'))
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')

        # --- NEW FEATURES IMPLEMENTATION ---
        
        elif operation == "Brightness":
            beta = params.get('beta', 0.0)
            img_float = processed_data.astype(float)
            processed_data = np.clip(img_float + beta, 0, 255)

        elif operation == "Contrast":
            alpha = params.get('alpha', 1.0)
            img_float = processed_data.astype(float)
            processed_data = np.clip(alpha * (img_float - 128) + 128, 0, 255)

        elif operation == "Exposure":
            stops = params.get('stops', 0.0)
            img_float = processed_data.astype(float)
            processed_data = np.clip(img_float * (2 ** stops), 0, 255)

        elif operation == "Saturation":
            scale = params.get('scale', 1.0)
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                img_float = processed_data[:, :, :3].astype(float)
                # Calculate grayscale luminance
                luminance = 0.299 * img_float[:,:,0] + 0.587 * img_float[:,:,1] + 0.114 * img_float[:,:,2]
                luminance = np.expand_dims(luminance, axis=2)
                # Blend luminance with original colors
                blended = luminance + scale * (img_float - luminance)
                processed_data[:, :, :3] = np.clip(blended, 0, 255)

        elif operation == "Warmth":
            warmth = params.get('warmth', 0.0)
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                img_float = processed_data[:, :, :3].astype(float)
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] + warmth, 0, 255) # Adjust Red
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] - warmth, 0, 255) # Adjust Blue
                processed_data[:, :, :3] = img_float

        elif operation == "Vignette":
            intensity = params.get('intensity', 0.5)
            img_float = processed_data.astype(float)
            h, w = img_float.shape[:2]
            # Create a grid of x,y coordinates
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h / 2, w / 2
            # Calculate distance from center for every pixel
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            # Create radial mask
            mask = 1 - (intensity * (distance / max_distance))
            mask = np.clip(mask, 0, 1)
            
            if img_float.ndim == 3:
                mask = np.expand_dims(mask, axis=2)
            processed_data = np.clip(img_float * mask, 0, 255)

        elif operation == "Sharpness":
            amount = params.get('amount', 1.0)
            img_float = processed_data.astype(float)
            # Define Laplacian kernel for edge enhancement
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) * amount
            # If amount is less than 1, blend with identity matrix to reduce effect
            if amount < 1.0:
                kernel = kernel + np.array([[0, 0, 0], [0, 1 - amount, 0], [0, 0, 0]])
            
            if img_float.ndim == 3 and img_float.shape[2] >= 3:
                for i in range(3):
                    processed_data[:,:,i] = np.clip(convolve(img_float[:,:,i], kernel, mode='reflect'), 0, 255)
            else:
                processed_data = np.clip(convolve(img_float, kernel, mode='reflect'), 0, 255)

        elif operation == "Noise Reduction":
            strength = params.get('strength', 1.0)
            # Create a uniform mean filter based on strength
            k_size = int(strength) * 2 + 1 # Ensures an odd-sized kernel (3x3, 5x5, etc.)
            kernel = np.ones((k_size, k_size)) / (k_size * k_size)
            img_float = processed_data.astype(float)
            
            if img_float.ndim == 3 and img_float.shape[2] >= 3:
                for i in range(3):
                    processed_data[:,:,i] = convolve(img_float[:,:,i], kernel, mode='reflect')
            else:
                processed_data = convolve(img_float, kernel, mode='reflect')

        # Ensure output data type is consistent
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data

