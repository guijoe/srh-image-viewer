# 
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
    
class ChromaticAberrationWidget(BaseParamsWidget):
    """A widget for Chromatic Aberration transformation."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Phase shift intensity:"))
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 0.05)
        self.intensity_spin.setValue(0.008)
        self.intensity_spin.setSingleStep(0.001)
        self.intensity_spin.setDecimals(3)
        layout.addWidget(self.intensity_spin)

        layout.addStretch()

    def get_params(self) ->dict:
        return {'intensity': self.intensity_spin.value()}
    
class LUT1DParamsWidget(BaseParamsWidget):
    """A widget to apply a cinematic contrast curve."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Contrast factor(S-Curve):"))
        self.contrast_spin = QDoubleSpinBox()
        self.contrast_spin.setRange(0.0, 2.0)
        self.contrast_spin.setValue(1.0) 
        self.contrast_spin.setSingleStep(0.1)
        layout.addWidget(self.contrast_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {'contrast_factor': self.contrast_spin.value()}
    
class TealOrangeParamsWidget(BaseParamsWidget):
    """A widget to apply the cinematic Teal & Orange look."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Intensity:"))
        self.intensity_spin = QDoubleSpinBox()
        self.intensity_spin.setRange(0.0, 1.0)
        self.intensity_spin.setValue(0.5)
        self.intensity_spin.setSingleStep(0.1)
        layout.addWidget(self.intensity_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {'intensity': self.intensity_spin.value()}
    
class BokehParamsWidget(BaseParamsWidget):
    """A widget to simulate a lens blur (Bokeh)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Bokeh Radius (Size):"))
        self.size_spin = QDoubleSpinBox()
        self.size_spin.setRange(1.0, 30.0)
        self.size_spin.setValue(5.0)
        layout.addWidget(self.size_spin)

        layout.addStretch()

    def get_params(self) -> dict:
        return {'size': self.size_spin.value()}
    
class FilmGrainParamsWidget(BaseParamsWidget):
    """A widget to add grain texture to the image."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Grain Intensity:"))
        self.grain_spin = QDoubleSpinBox()
        self.grain_spin.setRange(0.0, 0.5)
        self.grain_spin.setValue(0.1)
        self.grain_spin.setSingleStep(0.01)
        layout.addWidget(self.grain_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {'grain_intensity': self.grain_spin.value()}

class VignetteParamsWidget(BaseParamsWidget):
    """A widget to add a gradual darkness frame (Vignette)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Circle Radius:"))
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.1, 2.0)
        self.radius_spin.setValue(1.0)
        self.radius_spin.setSingleStep(0.1)
        layout.addWidget(self.radius_spin)

        layout.addWidget(QLabel("Softning (Falloff):"))
        self.falloff_spin = QDoubleSpinBox()
        self.falloff_spin.setRange(0.1, 5.0)
        self.falloff_spin.setValue(2.0)
        layout.addWidget(self.falloff_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'vignette_radius': self.radius_spin.value(),
            'vignette_falloff': self.falloff_spin.value()
        }

class ToneMappingParamsWidget(BaseParamsWidget):
    """A widget to simulate the High Dynamic Range Look (ACES)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Exposure (Brightness):"))
        self.exp_spin = QDoubleSpinBox()
        self.exp_spin.setRange(0.0, 5.0)
        self.exp_spin.setValue(1.0)
        self.exp_spin.setSingleStep(0.1)
        self.exp_spin.setDecimals(2)
        layout.addWidget(self.exp_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {'exposure': self.exp_spin.value()}  
    
class MotionBlurParamsWidget(BaseParamsWidget):
    """A widget to simulate motion blur."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Movement Length:"))
        self.len_spin = QDoubleSpinBox()
        self.len_spin.setRange(1, 50)
        self.len_spin.setValue(10)
        layout.addWidget(self.len_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'length': self.len_spin.value()}
    
class LensDistortionParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(QLabel("Distortion Type:"))
        self.distortion_type = QComboBox()
        self.distortion_type.addItems(["Barrel", "Pincushion"])
        layout.addWidget(self.distortion_type)
        
        layout.addWidget(QLabel("Distortion factor:"))
        self.factor_spin = QDoubleSpinBox()
        self.factor_spin.setRange(-1.0, 1.0)
        self.factor_spin.setValue(0.2)
        self.factor_spin.setSingleStep(0.05)
        layout.addWidget(self.factor_spin)
        
        layout.addWidget(QLabel("Scale factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 2.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setSingleStep(0.1)
        layout.addWidget(self.scale_spin)
        
        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'distortion_type': self.distortion_type.currentText(),
            'factor': self.factor_spin.value(),
            'scale': self.scale_spin.value()
        }

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
            "Gaussian Blur": GaussianParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Power Law (Gamma)": PowerLawParamsWidget,
            "Convolution": ConvolutionParamsWidget,
            "Chromatic Aberration (Cinema)": ChromaticAberrationWidget,
            "Cinematic 1D LUT": LUT1DParamsWidget,
            "Hollywood Teal & Orange": TealOrangeParamsWidget,
            "Cinematic Bokeh": BokehParamsWidget,
            "Film Grain (VFX)": FilmGrainParamsWidget,
            "Cinematic Vignette": VignetteParamsWidget,
            "ACES Tone Mapping": ToneMappingParamsWidget,
            "Motion Blur": MotionBlurParamsWidget,
            "Lens Distortion": LensDistortionParamsWidget,
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
        return "consuelo_cornejo Module"

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

        if operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            # skimage.filters.gaussian expects float data
            processed_data = skimage.filters.gaussian(processed_data.astype(float), sigma=sigma, preserve_range=True)
        elif operation == "Median Filter":
            filter_size = params.get('filter_size', 3)
            if filter_size <= 1: return processed_data # No change
            # skimage.filters.median
            if processed_data.ndim == 3 and processed_data.shape[2] in [3, 4]: # RGB/RGBA
                # Apply to each channel
                channels = []
                for i in range(processed_data.shape[2]):
                    channels.append(skimage.filters.median(processed_data[:,:,i], footprint=skimage.morphology.disk(int(filter_size/2))))
                processed_data = np.stack(channels, axis=-1)
            else:
                processed_data = skimage.filters.median(processed_data, footprint=skimage.morphology.disk(int(filter_size/2)))
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
        elif operation == "Chromatic Aberration (Cinema)":
            intensity = params.get('intensity', 0.008)

            if processed_data.ndim == 3 and processed_data.shape[2] >=3:
                h, w, chans = processed_data.shape
                y, x = np.indices((h, w))

                center_y, center_x = h / 2, w / 2
                ny, nx = (y - center_y) / center_y, (x - center_x) / center_x

                dist = np.sqrt(nx**2 + ny**2)

                scale_r = 1.0 + (intensity * dist)
                scale_b = 1.0 - (intensity * dist)

                def remap_channel(channel_idx, scale_map):
                    map_x = (nx * scale_map * center_x + center_x).clip(0, w - 1).astype(np.int32)
                    map_y = (ny * scale_map * center_y + center_y).clip(0, h - 1).astype(np.int32)
                    return processed_data[map_y, map_x, channel_idx]
                
                new_img = processed_data.copy()
                new_img[:, :, 0] = remap_channel(0, scale_r) 
                new_img[:, :, 2] = remap_channel(2, scale_b)
                processed_data =new_img
        elif operation == "Cinematic 1D LUT":
            factor = params.get('contrast_factor', 1.0)
            
            x = np.linspace(0, 1, 256)
            
            lut_curve = 1 / (1 + np.exp(-10 * factor * (x - 0.5)))
            
            lut_curve = ((lut_curve - lut_curve.min()) / (lut_curve.max() - lut_curve.min()) * 255).astype(np.uint8)

            if processed_data.dtype != np.uint8:
                processed_data = (processed_data * 255).astype(np.uint8)
            
            processed_data = lut_curve[processed_data]
        elif operation == "Hollywood Teal & Orange":
            intensity = params.get('intensity', 0.5)
            img_float = processed_data.astype(np.float32)/255.0

            x = np.linspace(0, 1, 256)
            s_curve = 1 / (1 + np.exp(-8*(x-0.5)))
            s_curve = (s_curve - s_curve.min()) / (s_curve.max()-s_curve.min())

            lut_r = np.clip(s_curve + (0.4*intensity*(x-0.5)), 0, 1)
            lut_g = s_curve
            lut_b = np.clip(s_curve - (0.6 * intensity * (x - 0.5)), 0, 1)

            lut_r = (lut_r * 255).astype(np.uint8)
            lut_g = (lut_g * 255).astype(np.uint8)
            lut_b = (lut_b * 255).astype(np.uint8)

            res = np.zeros_like(processed_data)
            res[:,:,0] = lut_r[processed_data[:,:,0]]
            res[:,:,1] = lut_g[processed_data[:,:,1]]
            res[:,:,2] = lut_b[processed_data[:,:,2]]

            processed_data = res
        elif operation == "Cinematic Bokeh":
            radius = int(params.get('size', 5))
            if radius <1: return processed_data

            kernel_size = 2*radius +1
            y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
            mask = x**2 + y**2 <= radius**2
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[mask]=1

            kernel /= kernel.sum()

            img_float = processed_data.astype(np.float32)
            if img_float.ndim == 3: 
                channels = []
                for i in range(img_float.shape[2]):
                    channel = img_float[:,:, i]
                    padded = np.pad(channel, radius, mode='reflect')
                    result = np.zeros_like(channel)

                    for r in range(channel.shape[0]):
                        for c in range(channel.shape[1]):
                            window = padded[r:r+kernel_size, c:c+kernel_size]
                            result[r, c] = np.sum(window * kernel)
                    channels.append(result)
                processed_data = np.stack(channels, axis =-1)
            else:
                padded = np.pad(img_float, radius, mode='reflect')
                result = np.zeros_like(img_float)
                for r in range(img_float.shape[0]):
                    for c in range(img_float.shape[1]):
                        window = padded[r:r+kernel_size, c:c+kernel_size]
                        result[r,c] = np.sum(window*kernel)
                processed_data = result
        elif operation == "Film Grain (VFX)":
            intensity = params.get('grain_intensity', 0.1)
            
            img_float = processed_data.astype(np.float32) / 255.0
            
            noise = np.random.normal(0, intensity, img_float.shape)
            
            luminance_mask = 4.0 * img_float * (1.0 - img_float)
            
            grainy_img = img_float + (noise * luminance_mask)
            processed_data = np.clip(grainy_img * 255, 0, 255).astype(np.uint8)
        elif operation == "Cinematic Vignette":
            radius_param = params.get('vignette_radius', 1.0)
            falloff = params.get('vignette_falloff', 2.0)
            
            rows, cols = processed_data.shape[:2]
            
            y, x = np.ogrid[-rows/2:rows/2, -cols/2:cols/2]
            
            max_dist = np.sqrt((rows/2)**2 + (cols/2)**2)
            dist = np.sqrt(x**2 + y**2) / (max_dist * radius_param)
            
            mask = np.clip(1.0 - (dist ** falloff), 0, 1)
            
            if processed_data.ndim == 3: 
                
                processed_data = (processed_data * mask[:, :, np.newaxis]).astype(np.uint8)
            else: 
                processed_data = (processed_data * mask).astype(np.uint8)
        elif operation == "ACES Tone Mapping":
            exposure = params.get('exposure', 1.0)
            img_f = (processed_data.astype(np.float32) / 255.0) * exposure
            
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            mapped = (img_f * (a * img_f + b)) / (img_f * (c * img_f + d) + e)
            
            processed_data = (np.clip(mapped, 0, 1) * 255).astype(np.uint8)
        elif operation == "Motion Blur":
            length = int(params.get('length', 10))
            if length <= 1:
                processed_data = image_data
            else:
                img_f = image_data.astype(np.float32)
                h, w = img_f.shape[:2]
                
                acc = np.zeros_like(img_f)
                
                for i in range(length):
                    shift = i - (length // 2)
                    
                    shifted_img = np.roll(img_f, shift, axis=1)
                    
                    acc += shifted_img
                
                result = acc / length
                
                processed_data = np.clip(result, 0, 255).astype(np.uint8)
        elif operation == "Lens Distortion":
            distortion_type = params.get('distortion_type', "Barrel")
            factor = params.get('factor', 0.2)
            scale = params.get('scale', 1.0)
    
            if distortion_type == "Barrel":
                k = factor  
            else:  
                k = -factor
            h, w = image_data.shape[:2]
            cx, cy = w/2, h/2
            y, x = np.indices((h,w))

            x_norm = (x - cx) / cx
            y_norm = (y - cy) / cy
    
            r = np.sqrt(x_norm**2 + y_norm**2)

            distortion_factor = 1 + k * r**2
            x_distorted = x_norm * distortion_factor
            y_distorted = y_norm * distortion_factor

            x_distorted = x_distorted * scale
            y_distorted = y_distorted * scale
    
            x_map = (x_distorted * cx + cx).astype(np.float32)
            y_map = (y_distorted * cy + cy).astype(np.float32)

            valid_mask = (x_map >=0) & (x_map < w) & (y_map >= 0) & (y_map < h)
    
            x_map = np.clip(x_map, 0, w - 1)
            y_map = np.clip(y_map, 0, h - 1)
    
            def remap_channel(channel, x_map, y_map):
                x0 = np.floor(x_map).astype(np.int32)
                x1 = x0 + 1
                y0 = np.floor(y_map).astype(np.int32)
                y1 = y0 + 1
        
                x0 = np.clip(x0, 0, w - 1)
                x1 = np.clip(x1, 0, w - 1)
                y0 = np.clip(y0, 0, h - 1)
                y1 = np.clip(y1, 0, h - 1)
        
                wa = (x1 - x_map) * (y1 - y_map)
                wb = (x_map - x0) * (y1 - y_map)
                wc = (x1 - x_map) * (y_map - y0)
                wd = (x_map - x0) * (y_map - y0)
        
                result = (wa * channel[y0, x0] +
                          wb * channel[y1, x0] +
                          wc * channel[y0, x1] +
                          wd * channel[y1, x1])
        
                return result
    
            if image_data.ndim == 3:
                result = np.zeros_like(image_data, dtype=np.float32)
                img_float = image_data.astype(np.float32)
                for i in range(image_data.shape[2]):
                    result[:, :, i] = remap_channel(img_float[:, :, i], x_map, y_map)
                for i in range(image_data.shape[2]):
                    result[~valid_mask, i] = 0
                processed_data = np.clip(result, 0, 255).astype(np.uint8)
            else:
                img_float = image_data.astype(np.float32)
                result = remap_channel(img_float, x_map, y_map)
                result[~valid_mask] = 0
                processed_data = np.clip(result, 0, 255).astype(np.uint8)
        
        processed_data = processed_data.astype(image_data.dtype)
        return processed_data