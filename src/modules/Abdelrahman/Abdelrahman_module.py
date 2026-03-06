from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from skimage import exposure, filters
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
class ThresholdParamsWidget(BaseParamsWidget):
    """A widget for simple binary thresholding."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Threshold Value (0-255):"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(127)
        layout.addWidget(self.slider)
        
        # Display the current value for better UX
        self.value_label = QLabel(f"Current: {self.slider.value()}")
        self.slider.valueChanged.connect(lambda v: self.value_label.setText(f"Current: {v}"))
        layout.addWidget(self.value_label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'threshold': self.slider.value()}

# Note: Image Inversion doesn't need a new class 
# we can just use NoParamsWidget since it has no settings.
# Define a custom control widget
class AbdelrahmanControlsWidget(QWidget):
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
            "Geometric": GeometricParamsWidget,
            "Invert (Negative)": NoParamsWidget,       # Added this
            "Binary Threshold": ThresholdParamsWidget, # Added this
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
class GeometricParamsWidget(BaseParamsWidget):
    """Widget for flipping and rotating images."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Transformation Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Horizontal Flip", 
            "Vertical Flip", 
            "Rotate 90° CW", 
            "Rotate 90° CCW", 
            "Rotate 180°",
            "Transpose"
        ])
        layout.addWidget(self.type_combo)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'transform_type': self.type_combo.currentText()}

class AbdelrahmanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Abdelrahman Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AbdelrahmanControlsWidget(module_manager, parent)
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
        # 1. Safety Check: If there's no data, return nothing to avoid crashes
        if image_data is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)

        # 2. Initialization
        processed_data = image_data.astype(np.float32)
        operation = params.get('operation')
        
        # 3. Axis Detection (Fixes the "Thin Line" issue)
        # Check if we have (1, H, W) or (H, W, 3)
        if processed_data.ndim == 3 and processed_data.shape[0] == 1:
            h_ax, w_ax = 1, 2
            c_ax = 0
        elif processed_data.ndim == 3:
            h_ax, w_ax = 0, 1
            c_ax = 2
        else:
            h_ax, w_ax = 0, 1
            c_ax = None

        # 4. Operations
        if operation == "Invert (Negative)":
            processed_data = 255.0 - processed_data

        elif operation == "Geometric":
            t_type = params.get('transform_type')
            if t_type == "Horizontal Flip":
                processed_data = np.flip(processed_data, axis=w_ax)
            elif t_type == "Vertical Flip":
                processed_data = np.flip(processed_data, axis=h_ax)
            elif t_type == "Rotate 90° CW":
                processed_data = np.rot90(processed_data, k=-1, axes=(h_ax, w_ax))
            elif t_type == "Rotate 90° CCW":
                processed_data = np.rot90(processed_data, k=1, axes=(h_ax, w_ax))
            elif t_type == "Rotate 180°":
                processed_data = np.rot90(processed_data, k=2, axes=(h_ax, w_ax))

        elif operation == "Binary Threshold":
            threshold_val = params.get('threshold', 127)
            # Convert to gray for the math
            if c_ax is not None:
                gray = np.mean(processed_data, axis=c_ax)
            else:
                gray = processed_data
            
            binary = (gray > threshold_val).astype(np.float32) * 255.0
            
            # Restore original dimensions so Napari doesn't crash
            if processed_data.ndim == 3:
                if processed_data.shape[0] == 1:
                    processed_data = binary[np.newaxis, :, :]
                else:
                    processed_data = np.stack([binary] * 3, axis=-1)
            else:
                processed_data = binary

        # 5. FINAL RETURN (Must be outside all if/elif blocks)
        return np.clip(processed_data, 0, 255).astype(np.uint8)