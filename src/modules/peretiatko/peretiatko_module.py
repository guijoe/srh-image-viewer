from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QStackedWidget, QDoubleSpinBox
from PySide6.QtCore import Signal
import numpy as np
import imageio
from scipy.ndimage import rotate

from modules.i_image_module import IImageModule


# --- Parameter Widgets ---

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


class RotateParamsWidget(BaseParamsWidget):
    """A widget for rotation angle parameter."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Angle (degrees):"))
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setMinimum(-360.0)
        self.angle_spinbox.setMaximum(360.0)
        self.angle_spinbox.setValue(90.0)
        self.angle_spinbox.setSingleStep(1.0)
        layout.addWidget(self.angle_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'angle': self.angle_spinbox.value()}


# --- Control Widget ---

class PeretiatkoControlsWidget(QWidget):
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
            "Rotate": RotateParamsWidget,
            "Flip Horizontal": NoParamsWidget,
            "Flip Vertical": NoParamsWidget,
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


# --- Module Class ---

class PeretiatkoImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Peretiatko Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "gif", "tif", "tiff"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = PeretiatkoControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                pass
            elif image_data.ndim == 2:
                pass
            else:
                print(f"Warning: Unexpected image dimensions {image_data.shape}")

            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        operation = params.get('operation')

        if operation == "Rotate":
            angle = params.get('angle', 90.0)
            if image_data.ndim == 3 and image_data.shape[2] in [3, 4]:
                # Rotate in the spatial (H, W) plane only, not the channel axis
                rotated = rotate(image_data, angle, axes=(0, 1), reshape=False)
            else:
                rotated = rotate(image_data, angle, reshape=False)
            # Clip to valid range for the dtype and cast back
            if np.issubdtype(image_data.dtype, np.integer):
                info = np.iinfo(image_data.dtype)
                rotated = np.clip(rotated, info.min, info.max)
            return rotated.astype(image_data.dtype)

        elif operation == "Flip Horizontal":
            return np.flip(image_data, axis=1).astype(image_data.dtype)

        elif operation == "Flip Vertical":
            return np.flip(image_data, axis=0).astype(image_data.dtype)

        return image_data.copy()
