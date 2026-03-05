from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox, QGridLayout
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
  
class WarmCoolParamsWidget(BaseParamsWidget):
    """Widget for warm/cool"""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Temperature (-100 = cool, +100 = warm):"))
        self.temp_spinbox = QDoubleSpinBox()
        self.temp_spinbox.setMinimum(-100.0)
        self.temp_spinbox.setMaximum(100.0)
        self.temp_spinbox.setValue(50.0)
        self.temp_spinbox.setSingleStep(5.0)
        layout.addWidget(self.temp_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'temperature': self.temp_spinbox.value()}
    
class VignetteParamsWidget(BaseParamsWidget):
    """Widget for Vignette effect."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Strength (0-1):"))
        self.strength_spinbox = QDoubleSpinBox()
        self.strength_spinbox.setMinimum(0.0)
        self.strength_spinbox.setMaximum(1.0)
        self.strength_spinbox.setValue(0.5)
        self.strength_spinbox.setSingleStep(0.05)
        layout.addWidget(self.strength_spinbox)

        layout.addWidget(QLabel("Radius (0-1):"))
        self.radius_spinbox = QDoubleSpinBox()
        self.radius_spinbox.setMinimum(0.1)
        self.radius_spinbox.setMaximum(1.0)
        self.radius_spinbox.setValue(0.7)
        self.radius_spinbox.setSingleStep(0.05)
        layout.addWidget(self.radius_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'strength': self.strength_spinbox.value(),
            'radius': self.radius_spinbox.value()
        }


class BleachBypassParamsWidget(BaseParamsWidget):
    """Widget for Bleach Bypass effect."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Bypass Amount (0-1):"))
        self.bypass_spinbox = QDoubleSpinBox()
        self.bypass_spinbox.setMinimum(0.0)
        self.bypass_spinbox.setMaximum(1.0)
        self.bypass_spinbox.setValue(0.5)
        self.bypass_spinbox.setSingleStep(0.05)
        layout.addWidget(self.bypass_spinbox)

        layout.addWidget(QLabel("Contrast Boost:"))
        self.contrast_spinbox = QDoubleSpinBox()
        self.contrast_spinbox.setMinimum(1.0)
        self.contrast_spinbox.setMaximum(3.0)
        self.contrast_spinbox.setValue(1.5)
        self.contrast_spinbox.setSingleStep(0.1)
        layout.addWidget(self.contrast_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'bypass': self.bypass_spinbox.value(),
            'contrast': self.contrast_spinbox.value()
        }

class KodakPortraParamsWidget(BaseParamsWidget):
    """Widget for Kodak Portra Film Emulation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Grain Strength (0-1):"))
        self.grain_spinbox = QDoubleSpinBox()
        self.grain_spinbox.setMinimum(0.0)
        self.grain_spinbox.setMaximum(1.0)
        self.grain_spinbox.setValue(0.3)
        self.grain_spinbox.setSingleStep(0.05)
        layout.addWidget(self.grain_spinbox)

        layout.addWidget(QLabel("Fade Amount (0-1):"))
        self.fade_spinbox = QDoubleSpinBox()
        self.fade_spinbox.setMinimum(0.0)
        self.fade_spinbox.setMaximum(1.0)
        self.fade_spinbox.setValue(0.2)
        self.fade_spinbox.setSingleStep(0.05)
        layout.addWidget(self.fade_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'grain': self.grain_spinbox.value(),
            'fade': self.fade_spinbox.value()
        }

class TiltShiftParamsWidget(BaseParamsWidget):
    """Widget for Tilt Shift effect."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Focus Center (0-1, 0=top 1=bottom):"))
        self.center_spinbox = QDoubleSpinBox()
        self.center_spinbox.setMinimum(0.0)
        self.center_spinbox.setMaximum(1.0)
        self.center_spinbox.setValue(0.5)
        self.center_spinbox.setSingleStep(0.05)
        layout.addWidget(self.center_spinbox)

        layout.addWidget(QLabel("Focus Band Width (0-1):"))
        self.width_spinbox = QDoubleSpinBox()
        self.width_spinbox.setMinimum(0.05)
        self.width_spinbox.setMaximum(0.8)
        self.width_spinbox.setValue(0.2)
        self.width_spinbox.setSingleStep(0.05)
        layout.addWidget(self.width_spinbox)

        layout.addWidget(QLabel("Blur Strength (sigma):"))
        self.blur_spinbox = QDoubleSpinBox()
        self.blur_spinbox.setMinimum(1.0)
        self.blur_spinbox.setMaximum(20.0)
        self.blur_spinbox.setValue(5.0)
        self.blur_spinbox.setSingleStep(1.0)
        layout.addWidget(self.blur_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'center': self.center_spinbox.value(),
            'width': self.width_spinbox.value(),
            'blur_sigma': self.blur_spinbox.value()
        }

class ColorFilterWidget(BaseParamsWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.set

# Define a custom control widget
class SampleControlsWidget(QWidget):
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
            "Warm/Cool Grading": WarmCoolParamsWidget,      
            "Vignette": VignetteParamsWidget,                  
            "Bleach Bypass": BleachBypassParamsWidget,        
            "Tilt Shift": TiltShiftParamsWidget,  
            "Kodak Portra": KodakPortraParamsWidget,            
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

class SampleImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "clara Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = SampleControlsWidget(module_manager, parent)
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
        if operation == "Warm/Cool Grading":
            temperature = params.get('temperature', 50.0)

            img_float = processed_data.astype(float)

            # Normalise temperature to -1 to +1 scale factor
            factor = temperature / 100.0

            # Each channel shifted linearly by factor * offset
            if img_float.ndim == 3 and img_float.shape[2] in [3, 4]:
                img_float[:, :, 0] = np.clip(img_float[:, :, 0] + factor * 30, 0, 255)  # Red
                img_float[:, :, 1] = np.clip(img_float[:, :, 1] + factor * 10, 0, 255)  # Green
                img_float[:, :, 2] = np.clip(img_float[:, :, 2] - factor * 30, 0, 255)  # Blue

            processed_data = img_float

        elif operation == "Vignette":
            strength = params.get('strength', 0.5)
            radius = params.get('radius', 0.7)

            img_float = processed_data.astype(float)
            h, w = img_float.shape[:2]

            # normalised coordinate grids centered at image center
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            X, Y = np.meshgrid(x, y)

            # Compute Euclidean distance from center for every pixel
            D = np.sqrt(X**2 + Y**2)

            #  Build vignette mask using Gaussian falloff
            # Pixels near center (D=0) get weight ~1 (unchanged)
            # Pixels near edges (D large) get weight ~0 (darkened)
            # sigma controls how quickly the darkening falls off
            sigma = radius / 2.0
            vignette_mask = np.exp(-(D**2) / (2 * sigma**2))

            # Blend mask with strength parameter
            # strength=0: no vignette, strength=1: full vignette
            vignette_mask = 1.0 - strength * (1.0 - vignette_mask)

            # Apply mask to each channel
            if img_float.ndim == 3:
                for i in range(img_float.shape[2]):
                    img_float[:, :, i] *= vignette_mask
            else:
                img_float *= vignette_mask

            processed_data = np.clip(img_float, 0, 255)

        elif operation == "Bleach Bypass":
            bypass = params.get('bypass', 0.5)
            contrast = params.get('contrast', 1.5)

            img_float = processed_data.astype(float) / 255.0

            # Convert to grayscale using weighted sum (ITU-R BT.601)
            if img_float.ndim == 3 and img_float.shape[2] in [3, 4]:
                luminance = (0.299 * img_float[:, :, 0] +
                            0.587 * img_float[:, :, 1] +
                            0.114 * img_float[:, :, 2])
                
                lum_3ch = np.stack([luminance, luminance, luminance], axis=2)
                desaturated = img_float * (1.0 - bypass) + lum_3ch * bypass

                contrasted = (desaturated - 0.5) * contrast + 0.5

                processed_data = np.clip(contrasted * 255.0, 0, 255)


        elif operation == "Tilt Shift":
            center = params.get('center', 0.5)
            band_width = params.get('width', 0.2)
            blur_sigma = params.get('blur_sigma', 5.0)

            img_float = processed_data.astype(float)
            h, w = img_float.shape[:2]

            # Create blurred version of entire image
            if img_float.ndim == 3 and img_float.shape[2] in [3, 4]:
                blurred = np.zeros_like(img_float)
                for i in range(img_float.shape[2]):
                    blurred[:, :, i] = skimage.filters.gaussian(
                        img_float[:, :, i], sigma=blur_sigma, preserve_range=True
                    )
            else:
                blurred = skimage.filters.gaussian(
                    img_float, sigma=blur_sigma, preserve_range=True
                )

            # Build a 1D gradient along the vertical axis
            # Pixels at center row get weight 1 
            # Pixels far from center get weight 0 
           
            # y_norm ranges 0 to 1 from top to bottom of image
            y_norm = np.linspace(0, 1, h)

            # Distance of each row from the focus center
            dist_from_center = np.abs(y_norm - center)

            # sigma of the focus band controls how sharp the transition is
            focus_band_sigma = band_width / 2.0
            focus_weights = np.exp(-(dist_from_center**2) / (2 * focus_band_sigma**2))

            # Reshape focus weights to broadcast across image width and channels
            if img_float.ndim == 3:
                focus_weights = focus_weights.reshape(h, 1, 1)
            else:
                focus_weights = focus_weights.reshape(h, 1)

            # Blend sharp and blurred using focus weights
            # output = sharp * focus_weight + blurred * (1 - focus_weight)
            result = img_float * focus_weights + blurred * (1.0 - focus_weights)

            processed_data = np.clip(result, 0, 255) 

        elif operation == "Kodak Portra":
            grain_strength = params.get('grain', 0.3)
            fade = params.get('fade', 0.2)

            img_float = processed_data.astype(float) / 255.0
           
            # Fade: model each channel with a different gamma curve
            # red gamma < 1 boosts reds, blue gamma > 1 crushes blues
            img_float[:, :, 0] = np.power(img_float[:, :, 0], 0.9)   # boost reds
            img_float[:, :, 1] = np.power(img_float[:, :, 1], 0.95)  # slight green boost
            img_float[:, :, 2] = np.power(img_float[:, :, 2], 1.1)   # crush blues slightly

            # lift blacks giving that soft faded film look
            # output = image * (1 - fade) + 0.5 * fade
            img_float = img_float * (1.0 - fade) + 0.5 * fade

            # Grain: Gaussian noise mimicking silver halide crystals
            if grain_strength > 0:
                grain = np.random.normal(0, grain_strength * 0.1, img_float.shape)
                img_float = np.clip(img_float + grain, 0, 1)

            processed_data = np.clip(img_float * 255.0, 0, 255)
                        
        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data