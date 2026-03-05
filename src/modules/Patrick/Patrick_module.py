from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio 
import skimage.filters
import skimage.morphology
from skimage.color import rgb2gray
from scipy.ndimage import convolve

from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

# Parameter Widgets 

class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
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
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Sigma (Standard Deviation):"))
        self.sigma_spinbox = QDoubleSpinBox()
        self.sigma_spinbox.setRange(0.1, 25.0)
        self.sigma_spinbox.setValue(1.0)
        layout.addWidget(self.sigma_spinbox)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'sigma': self.sigma_spinbox.value()}

class PowerLawParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma:"))
        self.gamma_spinbox = QDoubleSpinBox()
        self.gamma_spinbox.setRange(0.01, 5.0)
        self.gamma_spinbox.setValue(1.0)
        layout.addWidget(self.gamma_spinbox)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'gamma': self.gamma_spinbox.value()}

class ConvolutionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("3x3 Kernel:"))
        grid_layout = QGridLayout()
        self.kernel_inputs = []
        for r in range(3):
            row_inputs = []
            for c in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-100.0, 100.0)
                spinbox.setValue(1.0 if r == 1 and c == 1 else 0.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)
    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}

class DehazeParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Contrast Stretch (Clipping %):"))
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(0.0, 10.0)
        self.clip_spin.setValue(1.0)
        layout.addWidget(self.clip_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'clip_limit': self.clip_spin.value()}

class RunwayParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Edge Sensitivity (Threshold):"))
        self.thresh_spin = QDoubleSpinBox()
        self.thresh_spin.setRange(5.0, 200.0)
        self.thresh_spin.setValue(30.0)
        layout.addWidget(self.thresh_spin)
        layout.addStretch()
    def get_params(self) -> dict:
        return {'threshold': self.thresh_spin.value()}

# Control Widget

class PatrickControlsWidget(QWidget):
    process_requested = Signal(dict)
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Aviation Imaging Panel</h3>"))
        self.operation_selector = QComboBox()
        layout.addWidget(self.operation_selector)
        self.params_stack = QStackedWidget()
        layout.addWidget(self.params_stack)

        operations = {
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Aviation: Dehaze": DehazeParamsWidget,
            "Aviation: Horizon Detect": NoParamsWidget,
            "Aviation: Night Vision (EFVS)": NoParamsWidget,
            "Aviation: Runway Highlighting": RunwayParamsWidget,
            "Aviation: FOD Alert": NoParamsWidget
        }

        for name, widget_class in operations.items():
            widget = widget_class()
            self.params_stack.addWidget(widget)
            self.param_widgets[name] = widget
            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.operation_selector.currentTextChanged.connect(self._on_operation_changed)
        layout.addWidget(self.apply_button)

    def _on_apply_clicked(self):
        op_name = self.operation_selector.currentText()
        params = self.param_widgets[op_name].get_params()
        params['operation'] = op_name
        self.process_requested.emit(params)

    def _on_operation_changed(self, name):
        self.params_stack.setCurrentWidget(self.param_widgets[name])

# Main Module

class PatrickImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Patrick Aviation Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = PatrickControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def custom_numpy_convolve(self, image, kernel):
        """Pure NumPy convolution for course compliance."""
        i_h, i_w = image.shape
        k_h, k_w = kernel.shape
        pad_h, pad_w = k_h // 2, k_w // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(image)
        for y in range(i_h):
            for x in range(i_w):
                output[y, x] = np.sum(padded[y:y+k_h, x:x+k_w] * kernel)
        return output

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            return True, image_data, {'name': file_path.split('/')[-1]}, None
        except Exception as e:
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')

        # Original 3 Operations
        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        
        elif operation == "Sobel Edge Detect":
            gray = rgb2gray(processed_data[:,:,:3]) if processed_data.ndim == 3 else processed_data
            processed_data = skimage.filters.sobel(gray)
            
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                processed_data = np.power(input_float / max_val, gamma) * max_val

        elif operation == "Convolution":
            kernel = params.get('kernel')
            if kernel is not None:
                input_float = processed_data.astype(float)
                if input_float.ndim == 3:
                    channels = [convolve(input_float[:,:,i], kernel, mode='reflect') for i in range(input_float.shape[2])]
                    processed_data = np.stack(channels, axis=-1)
                else:
                    processed_data = convolve(input_float, kernel, mode='reflect')

        # 5 Aviation Operations (Pure NumPy)
        elif operation == "Aviation: Dehaze":
            clip = params.get('clip_limit', 1.0)
            low, high = np.percentile(processed_data, (clip, 100 - clip))
            processed_data = np.clip(processed_data, low, high)
            processed_data = (processed_data - low) / (high - low + 1e-6) * 255

        elif operation == "Aviation: Horizon Detect":
            gray = np.mean(processed_data, axis=2) if processed_data.ndim == 3 else processed_data
            h_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            processed_data = self.custom_numpy_convolve(gray.astype(float), h_kernel)

        elif operation == "Aviation: Night Vision (EFVS)":
            gray = np.mean(processed_data, axis=2) if processed_data.ndim == 3 else processed_data
            norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-6)
            red = norm * 255
            green = np.where(norm > 0.5, (norm - 0.5) * 510, 0)
            processed_data = np.stack([red, green, np.zeros_like(red)], axis=-1)

        elif operation == "Aviation: Runway Highlighting":
            thresh = params.get('threshold', 30.0)
            laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            if processed_data.ndim == 3:
                channels = [self.custom_numpy_convolve(processed_data[..., i].astype(float), laplacian) for i in range(3)]
                edges = np.abs(np.stack(channels, axis=-1))
            else:
                edges = np.abs(self.custom_numpy_convolve(processed_data.astype(float), laplacian))
            processed_data = np.where(edges > thresh, 255, 0)

        elif operation == "Aviation: FOD Alert":
            img_f = processed_data.astype(float)
            mask = np.abs(img_f - np.mean(img_f)) > (np.std(img_f) * 3)
            processed_data = np.where(mask, 255, 0)

        return processed_data.astype(np.uint8)