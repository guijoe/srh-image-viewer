from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QStackedWidget, QPushButton
from PySide6.QtCore import Signal
import numpy as np
import imageio

from modules.i_image_module import IImageModule

# ==============================================================================
# 1. PARAMETER WIDGETS
# ==============================================================================

class ContrastStretchingParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("New Min (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setRange(0.0, 255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)
        
        layout.addWidget(QLabel("New Max (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setRange(0.0, 255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'new_min': self.min_spinbox.value(), 'new_max': self.max_spinbox.value()}

class GammaParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Gamma Value (0.1 to 5.0):"))
        
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setRange(0.1, 5.0)
        self.gamma_spinbox.setSingleStep(0.1)
        self.gamma_spinbox.setValue(1.0) 
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}

class EmptyParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("No settings needed. Pure Math!"))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

# ==============================================================================
# 2. CONTROLS WIDGET
# ==============================================================================

class AdamYoussefControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Adam's Advanced Suite</h3>"))

        layout.addWidget(QLabel("Select Filter:"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)

        # QStackedWidget dynamically swaps the active parameter menu
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "1. Contrast Stretching": ContrastStretchingParamsWidget,
            "2. Gamma Correction": GammaParamsWidget,
            "3. Histogram Equalization": EmptyParamsWidget,
            "4. Edge Detection (Sobel)": EmptyParamsWidget,
            "5. Sepia Tone Matrix": EmptyParamsWidget
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
        params['operation'] = operation_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, operation_name: str):
        if operation_name in self.param_widgets:
            self.params_stack.setCurrentWidget(self.param_widgets[operation_name])

# ==============================================================================
# 3. MAIN MODULE CLASS
# ==============================================================================

class AdamYoussefImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Adam Youssef"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = AdamYoussefControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            # Normalize 2D grayscale shape into 3D expected format
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                image_data = image_data[np.newaxis, :]
            return True, image_data, {'name': file_path.split('/')[-1]}, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        # Work on a copy to avoid destructively altering the original data
        processed_data = image_data.copy()
        operation = params.get('operation')

        if operation == "1. Contrast Stretching":
            # Convert to float to prevent integer overflow
            img_float = processed_data.astype(float)
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)
            current_min = np.min(img_float)
            current_max = np.max(img_float)
            
            # Apply linear scaling formula
            if current_max != current_min:
                processed_data = (img_float - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min
                processed_data = np.clip(processed_data, new_min, new_max)

        elif operation == "2. Gamma Correction":
            # Normalize to [0,1] for power law transform
            img_float = processed_data.astype(float) / 255.0
            gamma = params.get('gamma', 1.0)
            
            if gamma > 0:
                processed_data = np.power(img_float, gamma) * 255.0
            processed_data = np.clip(processed_data, 0, 255)

        elif operation == "3. Histogram Equalization":
            img_int = np.clip(processed_data, 0, 255).astype(int)
            hist, _ = np.histogram(img_int.flatten(), 256, [0, 256])
            
            # Calculate CDF and scale across 0-255 spectrum
            cdf = hist.cumsum()
            cdf_masked = np.ma.masked_equal(cdf, 0)
            cdf_masked = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())
            cdf = np.ma.filled(cdf_masked, 0)
            
            # Map original pixels to new equalized values
            processed_data = cdf[img_int]

        elif operation == "4. Edge Detection (Sobel)":
            # Convert to grayscale (edges rely on luminance)
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                gray = np.dot(processed_data[..., :3], [0.2989, 0.5870, 0.1140])
            else:
                gray = processed_data.astype(float)

            # Pad array to avoid out-of-bounds errors with 3x3 kernel
            padded = np.pad(gray, 1, mode='edge')
            
            # Vectorized Convolution using array slicing (avoids slow nested loops)
            Gx = (padded[2:, 2:] + 2*padded[1:-1, 2:] + padded[:-2, 2:]) - \
                 (padded[2:, :-2] + 2*padded[1:-1, :-2] + padded[:-2, :-2])
                 
            Gy = (padded[2:, 2:] + 2*padded[2:, 1:-1] + padded[2:, :-2]) - \
                 (padded[:-2, 2:] + 2*padded[:-2, 1:-1] + padded[:-2, :-2])
                 
            # Combine gradients using Pythagorean theorem
            magnitude = np.sqrt(Gx**2 + Gy**2)
            
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255.0
                
            if processed_data.ndim == 3:
                processed_data[..., 0] = magnitude
                processed_data[..., 1] = magnitude
                processed_data[..., 2] = magnitude
            else:
                processed_data = magnitude
                
            processed_data = np.clip(processed_data, 0, 255)

        elif operation == "5. Sepia Tone Matrix":
            # Matrix multiplication to cross-mix RGB channels
            if processed_data.ndim == 3 and processed_data.shape[2] >= 3:
                img_float = processed_data[..., :3].astype(float)
                
                sepia_matrix = np.array([[0.393, 0.769, 0.189],
                                         [0.349, 0.686, 0.168],
                                         [0.272, 0.534, 0.131]])
                                         
                sepia_img = img_float.dot(sepia_matrix.T)
                processed_data[..., :3] = np.clip(sepia_img, 0, 255)
            else:
                img_float = processed_data.astype(float)
                r = img_float * 1.351
                g = img_float * 1.203
                b = img_float * 0.937
                processed_data = np.stack([np.clip(r,0,255), np.clip(g,0,255), np.clip(b,0,255)], axis=-1)

        return processed_data.astype(image_data.dtype)