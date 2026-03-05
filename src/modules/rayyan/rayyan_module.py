from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout, QSpinBox
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# Parameter Widgets for Different Operations
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
    
class PosterizationParamsWidget(BaseParamsWidget):
    """A widget for Posterization parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the number of color levels
        layout.addWidget(QLabel("Color Levels (2-256):"))
        self.levels_spinbox = QSpinBox()
        self.levels_spinbox.setMinimum(2)
        self.levels_spinbox.setMaximum(256)
        self.levels_spinbox.setValue(4)  # Default value
        layout.addWidget(self.levels_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'levels': self.levels_spinbox.value()
        }

class VignetteParamsWidget(BaseParamsWidget):
    """A widget for Vignette parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the vignette radius
        layout.addWidget(QLabel("Radius Factor (0.1 - 5.0):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(0.1)
        self.radius_spinbox.setMaximum(5.0)
        self.radius_spinbox.setSingleStep(0.1)
        self.radius_spinbox.setValue(1.5) # Default value
        layout.addWidget(self.radius_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'radius_factor': self.radius_spinbox.value()
        }
        
class SolarizationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Intensity Threshold (0-255):"))
        self.threshold_spinbox = QSpinBox()
        self.threshold_spinbox.setMinimum(0)
        self.threshold_spinbox.setMaximum(255)
        self.threshold_spinbox.setValue(128)
        layout.addWidget(self.threshold_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'threshold': self.threshold_spinbox.value()}

class ChromaticAberrationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Shift Amount (Pixels):"))
        self.shift_spinbox = QSpinBox()
        self.shift_spinbox.setMinimum(1)
        self.shift_spinbox.setMaximum(50)
        self.shift_spinbox.setValue(10)
        layout.addWidget(self.shift_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'shift': self.shift_spinbox.value()}

class PixelationParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Block Size (Pixels):"))
        self.block_spinbox = QSpinBox()
        self.block_spinbox.setMinimum(2)
        self.block_spinbox.setMaximum(100)
        self.block_spinbox.setValue(10)
        layout.addWidget(self.block_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'block_size': self.block_spinbox.value()}

# Define a custom control widget
class RayyanControlsWidget(QWidget):
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

        # 1. Dropdown menu to select operations
        layout.addWidget(QLabel("Operation:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # 2. Description label for the transformation
        self.operation_description_label = QLabel()
        self.operation_description_label.setWordWrap(True)
        self.operation_description_label.setText("<font color='grey'><i>Select an operation to see its description and parameters.</i></font>")
        layout.addWidget(self.operation_description_label)

        # 3. Stacked widget to hold the parameter UIs (like sliders/spinboxes)
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        # Define operations and their corresponding parameter widgets
        self.operations = {
            "Sepia Filter": NoParamsWidget,
            "Posterization": PosterizationParamsWidget,
            "Vignette": VignetteParamsWidget,
            "Solarization": SolarizationParamsWidget,
            "Chromatic Aberration": ChromaticAberrationParamsWidget,
            "Pixelation": PixelationParamsWidget,
        }

        self.operation_descriptions = {
            "Sepia Filter": "Applies a vintage, warm brownish-gold color tone. Enhances image mood and soften harsh digital lighting.",
            "Posterization": "Reduces the number of unique colors, creating a stylistic pop-art look. Also acts as image compression.",
            "Vignette": "Darkens image edges smoothly, guiding the viewer's eye to the center and reducing edge distractions.",
            "Solarization": "Inverts pixels brighter than a threshold. Dramatic enhancement for surreal artistic effects or salvaging overexposure.",
            "Chromatic Aberration": "Shifts color channels horizontally. Adds motion or cinematic, raw digital realism by simulating lens flaws.",
            "Pixelation": "Groups pixels into larger blocks. Enhances privacy through censoring or creates a retro aesthetic."
        }

        # Populates the operation selector combobox
        for name, widget_class in self.operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        # Apply processing button
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
        # Update parameters stacked widget
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

        if operation_name in self.operation_descriptions:
            description = self.operation_descriptions[operation_name]
            self.operation_description_label.setText(f"<font color='grey'><i>{description}</i></font>")
        else:
            self.operation_description_label.setText("<font color='grey'><i>Select an operation to see its description and parameters.</i></font>")

class RayyanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Rayyan Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = RayyanControlsWidget(module_manager, parent)
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
                pass
            elif image_data.ndim == 2: # Grayscale
                image_data = image_data[np.newaxis, :] # Add a channel dimension for consistency if desired
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None 
        except Exception as e:
            print(f"Error loading 2D image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "Sepia Filter":
            if len(processed_data.shape) == 3 and processed_data.shape[2] >= 3:
                img_float = processed_data[..., :3].astype(float)
                
                # The Sepia transformation matrix
                sepia_matrix = np.array([
                    [0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131]
                ])
                
                # Apply matrix multiplication to the color channels
                sepia_img = np.dot(img_float, sepia_matrix.T)
                processed_data[..., :3] = np.clip(sepia_img, 0, 255)

        elif operation == "Posterization":
            levels = params.get('levels', 4) 
            factor = 255.0 / (levels - 1)
            
            img_float = processed_data.astype(float)
            quantized = np.round(img_float / factor) * factor
            processed_data = np.clip(quantized, 0, 255)

        elif operation == "Vignette":
            radius_factor = params.get('radius_factor', 1.5)
            rows, cols = processed_data.shape[:2]
            
            # Create a meshgrid representing the X and Y coordinates
            X_result, Y_result = np.meshgrid(np.arange(cols), np.arange(rows))
            center_x, center_y = cols / 2, rows / 2
            
            # Calculate distance of each pixel from the center
            distance = np.sqrt((X_result - center_x)**2 + (Y_result - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Create the Gaussian mask
            sigma = max_dist / radius_factor
            mask = np.exp(-(distance**2) / (2 * sigma**2))
            
            # Ensure the mask can broadcast to 3 channels (RGB)
            if len(processed_data.shape) == 3:
                mask = mask[..., np.newaxis]
                
            img_float = processed_data.astype(float)
            vignette_img = img_float * mask
            processed_data = np.clip(vignette_img, 0, 255)
            
        elif operation == "Solarization":
            threshold = params.get('threshold', 128)
            # np.where works like an if/else statement for the entire array at once
            processed_data = np.where(processed_data > threshold, 255 - processed_data, processed_data)

        elif operation == "Chromatic Aberration":
            shift = params.get('shift', 10)
            if len(processed_data.shape) == 3 and processed_data.shape[2] >= 3:
                # Roll the Red channel (index 0) to the left
                processed_data[..., 0] = np.roll(processed_data[..., 0], -shift, axis=1)
                # Roll the Blue channel (index 2) to the right
                processed_data[..., 2] = np.roll(processed_data[..., 2], shift, axis=1)

        elif operation == "Pixelation":
            block_size = params.get('block_size', 10)
            h, w = processed_data.shape[:2]
            
            # Calculate working area to avoid out-of-bounds errors on the edges
            h_adj = (h // block_size) * block_size
            w_adj = (w // block_size) * block_size
            
            # 1. Slice the array using strides to get only the top-left pixel of every block
            small_img = processed_data[0:h_adj:block_size, 0:w_adj:block_size]
            
            # 2. Blow those pixels back up to full block size using np.repeat
            pixelated = np.repeat(np.repeat(small_img, block_size, axis=0), block_size, axis=1)
            
            # 3. Paste the pixelated area back onto the original image
            processed_data[0:h_adj, 0:w_adj] = pixelated

        # Ensure output data type matches the original image
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data