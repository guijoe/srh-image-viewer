from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
from PySide6.QtCore import Qt, Signal
import numpy as np
import imageio # For general image loading (can use Pillow too)


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


class ContrastStretchingParamsWidget(BaseParamsWidget):
    """A widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Input for the new minimum value
        layout.addWidget(QLabel("New Minimum Intensity (0-255):"))
        self.min_spinbox = QDoubleSpinBox()
        self.min_spinbox.setMinimum(0.0)
        self.min_spinbox.setMaximum(255.0)
        self.min_spinbox.setValue(0.0)
        layout.addWidget(self.min_spinbox)

        # Input for the new maximum value
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

class BrightnessParamsWidget(BaseParamsWidget):
    """A widget for Brightness Adjustment."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brightness Adjustment Value (-255 to 255):"))

        self.brightness_spinbox = QDoubleSpinBox()
        self.brightness_spinbox.setMinimum(-255.0)
        self.brightness_spinbox.setMaximum(255.0)
        self.brightness_spinbox.setValue(0.0)
        self.brightness_spinbox.setSingleStep(5.0)

        layout.addWidget(self.brightness_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'value': self.brightness_spinbox.value()
        }

class HistogramEqualizationParamsWidget(BaseParamsWidget):
    """Histogram Equalization (no parameters needed)."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel("Histogram Equalization has no parameters.")
        label.setStyleSheet("font-style: italic; color: gray;")

        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class MedianFilterParamsWidget(BaseParamsWidget):
    """Median Filter parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Kernel Size (odd number):"))

        self.kernel_spinbox = QDoubleSpinBox()
        self.kernel_spinbox.setMinimum(1)
        self.kernel_spinbox.setMaximum(15)
        self.kernel_spinbox.setValue(3)
        self.kernel_spinbox.setSingleStep(2)

        layout.addWidget(self.kernel_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'kernel_size': int(self.kernel_spinbox.value())}

class GaussianNoiseParamsWidget(BaseParamsWidget):
    """Gaussian Noise parameters."""

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Noise Standard Deviation:"))

        self.std_spinbox = QDoubleSpinBox()
        self.std_spinbox.setMinimum(1.0)
        self.std_spinbox.setMaximum(100.0)
        self.std_spinbox.setValue(20.0)
        self.std_spinbox.setSingleStep(1.0)

        layout.addWidget(self.std_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'std': self.std_spinbox.value()}

class GeometryParamsWidget(BaseParamsWidget):
    """Geometry transformations: arbitrary rotation, mirrors, and crop."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # --- Rotation ---
        layout.addWidget(QLabel("Rotate (degrees):"))
        self.rotate_spinbox = QDoubleSpinBox()
        self.rotate_spinbox.setMinimum(-360)
        self.rotate_spinbox.setMaximum(360)
        self.rotate_spinbox.setSingleStep(1)
        self.rotate_spinbox.setValue(0)
        layout.addWidget(self.rotate_spinbox)

        # --- Mirror options ---
        layout.addWidget(QLabel("Mirror Options:"))
        self.hflip_checkbox = QPushButton("Horizontal Mirror")
        self.hflip_checkbox.setCheckable(True)
        layout.addWidget(self.hflip_checkbox)

        self.vflip_checkbox = QPushButton("Vertical Mirror")
        self.vflip_checkbox.setCheckable(True)
        layout.addWidget(self.vflip_checkbox)

        self.dflip_checkbox = QPushButton("Diagonal Mirror")
        self.dflip_checkbox.setCheckable(True)
        layout.addWidget(self.dflip_checkbox)

        # --- Crop ---
        layout.addWidget(QLabel("Crop (x1, y1, x2, y2):"))
        self.crop_x1 = QDoubleSpinBox(); layout.addWidget(self.crop_x1)
        self.crop_y1 = QDoubleSpinBox(); layout.addWidget(self.crop_y1)
        self.crop_x2 = QDoubleSpinBox(); layout.addWidget(self.crop_x2)
        self.crop_y2 = QDoubleSpinBox(); layout.addWidget(self.crop_y2)

        layout.addStretch()

    def get_params(self) -> dict:
        """Return parameters dictionary for processing."""
        return {
            'rotate_deg': float(self.rotate_spinbox.value()),
            'hflip': self.hflip_checkbox.isChecked(),
            'vflip': self.vflip_checkbox.isChecked(),
            'dflip': self.dflip_checkbox.isChecked(),
            'crop': (
                int(self.crop_x1.value()),
                int(self.crop_y1.value()),
                int(self.crop_x2.value()),
                int(self.crop_y2.value())
            )
        }

class SaltPepperNoiseParamsWidget(BaseParamsWidget):
    """Add Salt & Pepper Noise to the image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(QLabel("Noise Probability (0 to 1):"))

        self.prob_spinbox = QDoubleSpinBox()
        self.prob_spinbox.setMinimum(0.0)
        self.prob_spinbox.setMaximum(1.0)
        self.prob_spinbox.setSingleStep(0.01)
        self.prob_spinbox.setValue(0.05)
        layout.addWidget(self.prob_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'prob': self.prob_spinbox.value()}

class LaplacianFilterParamsWidget(BaseParamsWidget):
    """Apply Laplacian Filter to the image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(QLabel("Scale Factor:"))

        self.scale_spinbox = QDoubleSpinBox()
        self.scale_spinbox.setMinimum(0.1)
        self.scale_spinbox.setMaximum(10.0)
        self.scale_spinbox.setSingleStep(0.1)
        self.scale_spinbox.setValue(1.0)
        layout.addWidget(self.scale_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'scale': self.scale_spinbox.value()}

class NegativeParamsWidget(BaseParamsWidget):
    """Invert image colors."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("This operation has no parameters."))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

# Define a custom control widget
class MananControlsWidget(QWidget):
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
            "Brightness": BrightnessParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Negative / Invert": NegativeParamsWidget,
            "Geometry": GeometryParamsWidget,
            "Contrast Stretching": ContrastStretchingParamsWidget,
            "Histogram Equalization": HistogramEqualizationParamsWidget,
            "Add Gaussian Noise": GaussianNoiseParamsWidget,
            "Salt & Pepper Noise": SaltPepperNoiseParamsWidget,
            "Laplacian Filter": LaplacianFilterParamsWidget,
            "Median Filter": MedianFilterParamsWidget,

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

class MananImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Manan Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif","tif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = MananControlsWidget(module_manager, parent)
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


        if operation == "Power Law (Gamma)":
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
        
        elif operation == "Contrast Stretching":
            # Ensure we are working with a floating point image for calculations
            img_float = processed_data.astype(float)

            # Get parameters from the UI
            new_min = params.get('new_min', 0.0)
            new_max = params.get('new_max', 255.0)

            # Get current image intensity range
            current_min = np.min(img_float)
            current_max = np.max(img_float)

            # Avoid division by zero if the image is flat
            if current_max == current_min:
                return processed_data # Return original image

            # Apply the linear stretching formula
            processed_data = (img_float - current_min) * \
                             ((new_max - new_min) / (current_max - current_min)) + new_min

            # Clip values to be safe, though the formula should handle it
            processed_data = np.clip(processed_data, new_min, new_max)

        elif operation == "Brightness":
            value = params.get('value', 30)

            processed_data = processed_data.astype(np.int32) + value
            processed_data = np.clip(processed_data, 0, 255)

        elif operation == "Histogram Equalization":

            # Convert to grayscale if RGB
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]:
                gray = np.mean(processed_data[:, :, :3], axis=2).astype(np.uint8)
            else:
                gray = processed_data.astype(np.uint8)

            # Compute histogram
            hist, bins = np.histogram(gray.flatten(), 256, [0, 256])

            # Compute CDF
            cdf = hist.cumsum()

            # Normalize CDF
            cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())

            # Map original intensities
            equalized = np.interp(gray.flatten(), bins[:-1], cdf_normalized)
            processed_data = equalized.reshape(gray.shape)

        elif operation == "Median Filter":
            kernel_size = params.get('kernel_size', 3)
            pad = kernel_size // 2

            # Fix singleton grayscale channel
            if processed_data.ndim == 3 and processed_data.shape[0] == 1:
                processed_data = processed_data[0]  # shape (H, W)

            img = processed_data.astype(float)

            # Grayscale
            if img.ndim == 2:
                padded = np.pad(img, pad, mode='reflect')
                output = np.zeros_like(img)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        region = padded[i:i+kernel_size, j:j+kernel_size]
                        output[i,j] = np.median(region)
                processed_data = np.clip(output, 0, 255).astype(np.uint8)

            # RGB / RGBA
            else:
                output = np.zeros_like(img)
                for c in range(img.shape[2]):
                    padded = np.pad(img[:,:,c], pad, mode='reflect')
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            region = padded[i:i+kernel_size, j:j+kernel_size]
                            output[i,j,c] = np.median(region)
                processed_data = np.clip(output, 0, 255).astype(np.uint8)
                
        elif operation == "Add Gaussian Noise":

            std = params.get('std', 20)
            mean = 0
                
            noise = np.random.normal(mean, std, processed_data.shape)
            processed_data = processed_data.astype(np.float32) + noise
            processed_data = np.clip(processed_data, 0, 255)

        elif operation == "Geometry":
            img = processed_data.copy()

            # --- Ensure image is at least 2D ---
            if img.ndim == 3 and img.shape[0] == 1:  # Convert (1,H,W) -> (H,W)
                img = img[0]

            # Save original shape info
            orig_shape = img.shape

            # --- Crop ---
            h, w = img.shape[:2]
            x1, y1, x2, y2 = params.get('crop', (0,0,w,h))
            x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
            y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
            if x2 > x1 and y2 > y1:
                img = img[y1:y2, x1:x2]

            # --- Diagonal flip ---
            if params.get('dflip', False):
                if img.ndim == 3:
                    img = np.transpose(img, (1,0,2))
                else:
                    img = np.transpose(img, (1,0))

            # --- Horizontal / Vertical flips ---
            if params.get('hflip', False):
                img = np.fliplr(img)
            if params.get('vflip', False):
                img = np.flipud(img)

            # --- Arbitrary rotation ---
            deg = params.get('rotate_deg', 0)
            if deg != 0:
                theta = np.deg2rad(deg)
                cos_t, sin_t = np.cos(theta), np.sin(theta)
                h, w = img.shape[:2]
                new_w = int(abs(w*cos_t) + abs(h*sin_t))
                new_h = int(abs(w*sin_t) + abs(h*cos_t))
                if img.ndim == 3:
                    rotated = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
                else:
                    rotated = np.zeros((new_h, new_w), dtype=img.dtype)

                cx, cy = w / 2, h / 2
                ncx, ncy = new_w / 2, new_h / 2

                ys, xs = np.indices((new_h, new_w))
                x_src = (xs - ncx) * cos_t + (ys - ncy) * sin_t + cx
                y_src = -(xs - ncx) * sin_t + (ys - ncy) * cos_t + cy

                x_src_int = np.round(x_src).astype(int)
                y_src_int = np.round(y_src).astype(int)
                mask = (x_src_int >= 0) & (x_src_int < w) & (y_src_int >= 0) & (y_src_int < h)

                if img.ndim == 3:
                    rotated[mask] = img[y_src_int[mask], x_src_int[mask]]
                else:
                    rotated[mask] = img[y_src_int[mask], x_src_int[mask]]

                img = rotated

            # --- Flatten singleton channel dimension if present (H,W,1) -> (H,W) ---
            if img.ndim == 3 and img.shape[2] == 1:
                img = img[:,:,0]

            # --- Prevent flat image from crashing Napari ---
            img = img.astype(float)
            if np.min(img) == np.max(img):
                img += 1e-6

            # --- Clip to valid range and restore dtype ---
            processed_data = np.clip(img, 0, 255).astype(processed_data.dtype)

        elif operation == "Salt & Pepper Noise":
            prob = params.get('prob', 0.05)
            output = processed_data.copy()
            rnd = np.random.rand(*processed_data.shape)
            output[(rnd < prob/2)] = 0      # Pepper
            output[(rnd >= prob/2) & (rnd < prob)] = 255  # Salt
            processed_data = output

        elif operation == "Laplacian Filter":
            scale = params.get('scale', 1.0)
            kernel = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])

            # Fix grayscale singleton channel if needed
            if processed_data.ndim == 3 and processed_data.shape[0] == 1:
                processed_data = processed_data[0]  # shape (H, W)

            img = processed_data.astype(float)

            # Grayscale
            if img.ndim == 2:
                padded = np.pad(img, 1, mode='reflect')
                output = np.zeros_like(img)
                for i in range(img.shape[0]):
                    for j in range(img.shape[1]):
                        output[i,j] = np.sum(padded[i:i+3, j:j+3] * kernel)
                processed_data = np.clip(img + scale * output, 0, 255).astype(np.uint8)

            # RGB / RGBA
            else:
                output = np.zeros_like(img)
                for c in range(img.shape[2]):
                    padded = np.pad(img[:,:,c], 1, mode='reflect')
                    for i in range(img.shape[0]):
                        for j in range(img.shape[1]):
                            output[i,j] = np.sum(padded[i:i+3, j:j+3] * kernel)
                    processed_data[:,:,c] = np.clip(img[:,:,c] + scale * output, 0, 255)
                processed_data = processed_data.astype(np.uint8)

        elif operation == "Negative / Invert":
            processed_data = 255 - processed_data


        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data