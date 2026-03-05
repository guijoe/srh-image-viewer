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

class Circle(BaseParamsWidget):
    """A widget for Circle parameters.

    The spinbox limits depend on the size of the current image.  Since the
    widget is created before an image is loaded we initialise the dimensions
    to zero and provide a helper method that the control panel can call when a
    new image arrives.  This avoids referencing "self.width" and
    "self.height" before they exist.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # these will be filled once an image is shown
        self.width = 0
        self.height = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

         # Input for the new y value
        layout.addWidget(QLabel("X Coordinates:"))
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setMinimum(0.0)
        self.y_spinbox.setMaximum(0.0)
        self.y_spinbox.setValue(-1.0)
        self.y_spinbox.setSingleStep(1.0)
        layout.addWidget(self.y_spinbox)

        # Input for the new x value
        layout.addWidget(QLabel("Y Coordinates:"))
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setMinimum(0.0)
        self.x_spinbox.setMaximum(0.0)
        self.x_spinbox.setValue(-1.0)
        self.x_spinbox.setSingleStep(1.0)
        layout.addWidget(self.x_spinbox)


        # Input for the new maximum value
        layout.addWidget(QLabel("Circle Radius:"))
        self.r_spinbox = QDoubleSpinBox()
        self.r_spinbox.setMinimum(0.0)
        self.r_spinbox.setMaximum(1000000.0) # Arbitrary large value, will be updated in ``set_image_size``
        self.r_spinbox.setValue(0.0)
        self.r_spinbox.setSingleStep(1.0)
        layout.addWidget(self.r_spinbox)

        layout.addWidget(QLabel("Shade width:"))
        self.a_spinbox = QDoubleSpinBox()
        self.a_spinbox.setMinimum(0.0)
        self.a_spinbox.setMaximum(10000.0) # Arbitrary large value
        self.a_spinbox.setValue(-1.0)
        self.a_spinbox.setSingleStep(10.0)
        layout.addWidget(self.a_spinbox)

        layout.addStretch()

    def set_image_size(self, width: int, height: int):
        """Update internal dimensions and adjust spinbox ranges.

        This should be called whenever a new image is loaded so that the user
        cannot enter coordinates outside the image bounds.  "width" refers to
        the number of columns and "height" to the rows.
        """
        self.width = width
        self.height = height

        # the circle centre coordinates use row/column ordering in the
        # processing code, so x corresponds to height (rows) and y to width
        self.x_spinbox.setMaximum(self.height)
        if self.x_spinbox.value() == 0.0:
            self.x_spinbox.setValue(self.height // 2)        
        self.x_spinbox.setSingleStep(self.height / 10)

        self.y_spinbox.setMaximum(self.width)
        if self.y_spinbox.value() == 0.0:
            self.y_spinbox.setValue(self.width // 2)
        self.y_spinbox.setSingleStep(self.width / 10)

        # choose a reasonable default radius (half the smaller dimension)
        if self.r_spinbox.value() == 0.0:
            self.r_spinbox.setValue(min(self.width, self.height) // 2)
        self.r_spinbox.setSingleStep(min(self.width, self.height) / 10)

        if self.a_spinbox.value() == 0.0: # only update if still at default
            self.a_spinbox.setValue(min(self.width, self.height) // 3)


    def get_params(self) -> dict:
        return {
            'xc': self.x_spinbox.value(),
            'yc': self.y_spinbox.value(),
            'r': self.r_spinbox.value(),
            'a': self.a_spinbox.value(),
        }
    
class Retro(BaseParamsWidget):
    """A widget for Contrast Stretching parameters."""
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

         # Input for the new y value
        layout.addWidget(QLabel("Number of Bits per Element:"))
        # use our custom spinbox so up/down arrows jump between powers of two
        self.nB_spinbox = QDoubleSpinBox()
        self.nB_spinbox.setMinimum(1)
        self.nB_spinbox.setMaximum(8)
        self.nB_spinbox.setValue(3.0)
        self.nB_spinbox.setSingleStep(1.0)
        layout.addWidget(self.nB_spinbox)

        # Input for the new x value
        layout.addWidget(QLabel("Factor for Resolution:"))
        self.fact_spinbox = QDoubleSpinBox()
        self.fact_spinbox.setMinimum(1.0)
        self.fact_spinbox.setMaximum(16.0)
        self.fact_spinbox.setValue(4.0)
        layout.addWidget(self.fact_spinbox)

        layout.addStretch()

    def get_params(self) -> dict:
        return {
            'fact': self.fact_spinbox.value(),
            'nB': self.nB_spinbox.value(),
        }

# Define a custom control widget
class Joaquin_SMSControlsWidget(QWidget):
    # Signal to request processing from the module manager
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

#'''
        # if the module manager is available we can listen for new images and
        # adjust parameter ranges accordingly
        if self.module_manager is not None:
            self.module_manager.image_loaded_and_processed.connect(
                self._on_image_loaded)

    def _on_image_loaded(self, image_data, metadata, session_id):
        """Called when the module manager loads or processes an image.

        We extract the width/height and pass them to any parameter widget that
        supports "set_image_size".  This ensures spinboxes always have valid
        bounds.
        """
        # image_data can be 2D or 3D (channels).  take the first two dims.
        if image_data is None:
            return
        try:
            height, width = image_data.shape[:2]
        except Exception:
            # unexpected shape, ignore
            return

        for widget in self.param_widgets.values():
            if hasattr(widget, 'set_image_size'):
                widget.set_image_size(width, height)
#'''
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
            "Shaded_Circle": Circle,
            "Retro_Pixelation": Retro,
            
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

class Joaquin_SMSImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Joaquin_SMS Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"] # Common formats

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = Joaquin_SMSControlsWidget(module_manager, parent)
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

        if operation == "Shaded_Circle":
            # Normalize to [0, 1]
            input_float = processed_data.astype(float)

            r = params.get('r',  input_float.shape[1] // 2)
            y = params.get('xc', input_float.shape[0] // 2)
            x = params.get('yc', input_float.shape[1] // 2)
            a = params.get('a', input_float.shape[1] // 3)
            
            feather = params.get("feather", a)  # how soft the edge is (pixels)

            # coordinate grids
            rows, cols, _ = input_float.shape
            yy, xx = np.ogrid[:rows, :cols]
            dist = np.sqrt(((rows - yy) - y)**2 + (xx - x)**2)

            # alpha outside the circle with a smooth ramp:
            # dist <= r        -> alpha = 0
            # dist >= r+feather-> alpha = 1
            alpha = np.clip((dist - r) / feather, 0.0, 1.0).astype(np.float32)

            # optional: make it smoother (ease curve)
            alpha = (alpha ** 2) * (3 - 2 * alpha)  # smoothstep

            overlay = np.array([25, 25, 30], dtype=np.float32)  # your tint color     | e.g. [170, 140, 90]

            # blend
            alpha3 = alpha[..., None]  # (H, W, 1) so it applies to RGB
            out = (1.0 - alpha3) * input_float + alpha3 * overlay

            # back to original dtype range
            processed_data = np.clip(out, 0, 255).astype(image_data.dtype)

        elif operation == "Retro_Pixelation":
            # simple pixel‑ation effect controlled by two parameters:
            # ``nB`` = number of quantization bits, ``fact`` = downsampling factor.
            input_float = processed_data.astype(float)/ 255.0

            num_bits = int(params.get('nB', 3)) # Number of bits per channel for quantization
            factor = int(params.get('fact', 4)) # Downsampling factor (how much to reduce resolution before upscaling)
            if factor < 1:
                factor = 1

            # downsample every `factor` pixels (channels preserved)
            small = input_float[::factor, ::factor]

            # quantise in [0,1]
            levels = 2 ** num_bits
            small = np.round(small * (levels - 1)) / (levels - 1)

            # upsample back
            up = np.repeat(np.repeat(small, factor, axis=0), factor, axis=1)
            rows, cols = input_float.shape[:2]
            processed_data = up[:rows, :cols] * 255.0
            # later the dtype cast will clip/convert to uint8
        
        # Ensure output data type is consistent (e.g., convert back to uint8 if processing changed it)
        processed_data = processed_data.astype(image_data.dtype)

        return processed_data