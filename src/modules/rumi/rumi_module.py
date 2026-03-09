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
                if r == 1 and c == 1:
                    spinbox.setValue(1.0)
                grid_layout.addWidget(spinbox, r, c)
                row_inputs.append(spinbox)
            self.kernel_inputs.append(row_inputs)
        layout.addLayout(grid_layout)

    def get_params(self) -> dict:
        kernel = np.array([[spinbox.value() for spinbox in row] for row in self.kernel_inputs])
        return {'kernel': kernel}

class ContrastStretchingParamsWidget(BaseParamsWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)
        layout.addWidget(QLabel("New Maximum Intensity (0-255):"))
        self.max_spinbox = QDoubleSpinBox()
        self.max_spinbox.setMinimum(0.0)
        self.max_spinbox.setMaximum(255.0)
        self.max_spinbox.setValue(255.0)
        layout.addWidget(self.max_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'new_min': self.min_spinbox.value(),
            'new_max': self.max_spinbox.value()
        }


class FrequencyFilterParamsWidget(BaseParamsWidget):
   
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Filter Type:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Ideal Lowpass", "Gaussian Lowpass", "Ideal Highpass"])
        layout.addWidget(self.filter_combo)
        layout.addWidget(QLabel("Cutoff Frequency (D0):"))
        self.cutoff_spinbox = QDoubleSpinBox()
        self.cutoff_spinbox.setMinimum(1.0)
        self.cutoff_spinbox.setMaximum(100.0)
        self.cutoff_spinbox.setValue(30.0)
        self.cutoff_spinbox.setSingleStep(1.0)
        layout.addWidget(self.cutoff_spinbox)
        layout.addStretch()
    
    def get_params(self) -> dict:
        return {
            'filter_type': self.filter_combo.currentText(),
            'cutoff': self.cutoff_spinbox.value()
        }


class RumiControlsWidget(QWidget):
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
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Laplacian Sharpening": NoParamsWidget,
            "Frequency Domain Filter": FrequencyFilterParamsWidget,
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

class RumiImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Rumi Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = RumiControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)
#extremely difficult have to revise
#look up scripy doc again
    def apply_frequency_filter(self, image, filter_type, cutoff):
       
        from scipy.fft import fft2, fftshift, ifft2, ifftshift
        
        if image.dtype != np.float64:
            image = image.astype(np.float64)
        
        if len(image.shape) == 3:
            filtered_channels = []
            for channel in range(image.shape[2]):
                channel_data = image[:, :, channel]
                filtered_channel = self._process_single_channel(channel_data, filter_type, cutoff)
                filtered_channels.append(filtered_channel)
            filtered = np.stack(filtered_channels, axis=-1)
        else:
            filtered = self._process_single_channel(image, filter_type, cutoff)
        
        return filtered

    def _process_single_channel(self, channel_data, filter_type, cutoff):
       
        from scipy.fft import fft2, fftshift, ifft2, ifftshift
        
        M, N = channel_data.shape
        F = fft2(channel_data)
        F_shifted = fftshift(F)
        u = np.arange(M) - M // 2
        v = np.arange(N) - N // 2
        V, U = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        
        if filter_type == "Ideal Lowpass":
            H = (D <= cutoff).astype(float)
        elif filter_type == "Gaussian Lowpass":
            H = np.exp(-(D**2) / (2 * cutoff**2))
        elif filter_type == "Ideal Highpass":
            H = (D > cutoff).astype(float)
        else:
            H = np.ones_like(D)
        
        G_shifted = F_shifted * H
        G = ifftshift(G_shifted)
        filtered = np.real(ifft2(G))
        
        return filtered

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
            if filter_size <= 1:
                return processed_data
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
        elif operation == "Contrast Stretching":
            img_float = processed_data.astype(float)
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)
            current_min = np.min(img_float)
            current_max = np.max(img_float)
            if current_max == current_min:
                return processed_data
            processed_data = (img_float - current_min) * ((new_max - new_min) / (current_max - current_min)) + new_min
            processed_data = np.clip(processed_data, new_min, new_max)
        elif operation == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            input_float = processed_data.astype(float)
            max_val = np.max(input_float)
            if max_val > 0:
                normalized = input_float / max_val
                gamma_corrected = np.power(normalized, gamma)
                processed_data = gamma_corrected * max_val
      
        elif operation == "Frequency Domain Filter":
            filter_type = params.get('filter_type', 'Gaussian Lowpass')
            cutoff = params.get('cutoff', 30.0)
            filtered = self.apply_frequency_filter(processed_data, filter_type, cutoff)
            filtered = np.clip(filtered, 0, 255)
            processed_data = filtered.astype(image_data.dtype)
        elif operation == "Laplacian Sharpening":
            laplacian_kernel = np.array([[0, 1, 0],
                                         [1, -4, 1],
                                         [0, 1, 0]])
            if len(processed_data.shape) == 3:
                if processed_data.shape[2] == 3:
                    gray = 0.299 * processed_data[:,:,0] + 0.587 * processed_data[:,:,1] + 0.114 * processed_data[:,:,2]
                else:
                    gray = processed_data[:,:,0]
            else:
                gray = processed_data
            from scipy import signal
            laplacian = signal.convolve2d(gray, laplacian_kernel, mode='same', boundary='symm')
            sharpened = gray - laplacian
            sharpened = np.clip(sharpened, 0, 255)
            if len(processed_data.shape) == 3:
                processed_data = np.stack([sharpened]*3, axis=-1)
            else:
                processed_data = sharpened
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

        processed_data = processed_data.astype(image_data.dtype)
        return processed_data
#really bad for big images mayve can  be altered
#tested with 6 images
