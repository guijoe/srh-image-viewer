
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QStackedWidget, QDoubleSpinBox
)
from PySide6.QtCore import Signal
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from modules.i_image_module import IImageModule


# ---------------- Parameter Widgets ----------------
class BaseParamsWidget(QWidget):
    def get_params(self) -> dict:
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel("This operation has no parameters.")
        layout.addWidget(label)
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class GaussianParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Sigma (0.5–5.0):"))

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setMinimum(0.5)
        self.sigma_spin.setMaximum(5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)

        layout.addWidget(self.sigma_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma_spin.value())}


class UnsharpParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Sigma for blur (0.5–5.0):"))

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setMinimum(0.5)
        self.sigma_spin.setMaximum(5.0)
        self.sigma_spin.setSingleStep(0.1)
        self.sigma_spin.setValue(1.0)

        layout.addWidget(self.sigma_spin)
        layout.addStretch()

    def get_params(self) -> dict:
        return {"sigma": float(self.sigma_spin.value())}


# ---------------- Controls Widget ----------------
class DanielControlsWidget(QWidget):

    apply_clicked = Signal(str, dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)

        self.module_manager = module_manager

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select Operation"))

        self.operation_combo = QComboBox()
        layout.addWidget(self.operation_combo)

        self.param_stack = QStackedWidget()
        layout.addWidget(self.param_stack)

        self.apply_button = QPushButton("Apply")
        layout.addWidget(self.apply_button)

        layout.addStretch()

        self._setup_operations()

        self.operation_combo.currentIndexChanged.connect(
            self.param_stack.setCurrentIndex
        )

        self.apply_button.clicked.connect(self._apply_operation)

    def _setup_operations(self):

        self.operations = [
            ("Gaussian Blur", GaussianParamsWidget()),
            ("Unsharp Mask", UnsharpParamsWidget()),
            ("Median Filter", NoParamsWidget()),
        ]

        for name, widget in self.operations:
            self.operation_combo.addItem(name)
            self.param_stack.addWidget(widget)

    def _apply_operation(self):

        index = self.operation_combo.currentIndex()

        operation_name, widget = self.operations[index]
        params = widget.get_params()

        self.apply_clicked.emit(operation_name, params)


# ---------------- Image Processing ----------------
def gaussian_kernel(size, sigma):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel


def apply_convolution(image, kernel):

    k = kernel.shape[0]
    pad = k // 2

    padded = np.pad(image, pad, mode='reflect')

    windows = sliding_window_view(padded, (k, k))
    result = np.einsum('ijkl,kl->ij', windows, kernel)

    return result


def gaussian_blur(image, sigma):

    size = int(6 * sigma) + 1
    if size % 2 == 0:
        size += 1

    kernel = gaussian_kernel(size, sigma)
    return apply_convolution(image, kernel)


def unsharp_mask(image, sigma):

    blurred = gaussian_blur(image, sigma)
    sharpened = image + (image - blurred)

    return np.clip(sharpened, 0, 255)


def median_filter(image):

    windows = sliding_window_view(image, (3, 3))
    med = np.median(windows, axis=(-2, -1))

    padded = np.pad(med, 1, mode="edge")
    return padded


# ---------------- Main Module ----------------
class DanielImageModule(IImageModule):

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Daniel Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "tif", "tiff"]

    def get_controls_widget(self, module_manager, parent=None):

        if self._controls_widget is None:
            self._controls_widget = DanielControlsWidget(module_manager, parent)
            self._controls_widget.apply_clicked.connect(self.apply_operation)

        return self._controls_widget

    def apply_operation(self, operation, params):

        image = self.module_manager.image_data_store.current_image_data

        if image is None:
            return

        image = image.astype(np.float32)

        if operation == "Gaussian Blur":
            result = gaussian_blur(image, params["sigma"])

        elif operation == "Unsharp Mask":
            result = unsharp_mask(image, params["sigma"])

        elif operation == "Median Filter":
            result = median_filter(image)

        else:
            return

        result = result.astype(np.uint8)

        self.module_manager.image_data_store.set_image(result)
