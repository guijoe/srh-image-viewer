import numpy as np
import imageio

from skimage import filters, color

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from PySide6.QtCore import Signal

from modules.i_image_module import IImageModule


# -----------------------------
# CONTROL PANEL
# -----------------------------

class AsserAttiaControls(QWidget):

    process_requested = Signal(dict)

    def __init__(self, module_manager=None, parent=None):
        super().__init__(parent)

        self.module_manager = module_manager

        layout = QVBoxLayout(self)

        title = QLabel("Asser Attia Image Filters")
        layout.addWidget(title)

        layout.addWidget(QLabel("Choose Filter"))

        self.operation_selector = QComboBox()
        self.operation_selector.addItems([
            "Gaussian Blur",
            "Edge Detection"
        ])

        layout.addWidget(self.operation_selector)

        apply_button = QPushButton("Apply Filter")
        apply_button.clicked.connect(self.apply_filter)

        layout.addWidget(apply_button)

    def apply_filter(self):

        params = {
            "operation": self.operation_selector.currentText()
        }

        self.process_requested.emit(params)


# -----------------------------
# MODULE CLASS
# -----------------------------

class AsserAttiaModule(IImageModule):

    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self):
        return "Asser Attia Module"

    def get_supported_formats(self):
        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None):

        if self._controls_widget is None:

            self._controls_widget = AsserAttiaControls(
                module_manager,
                parent
            )

            self._controls_widget.process_requested.connect(
                self._handle_processing_request
            )

        return self._controls_widget

    def _handle_processing_request(self, params):

        if hasattr(self._controls_widget, "module_manager"):

            manager = self._controls_widget.module_manager

            if manager:
                manager.apply_processing_to_current_image(params)

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------

    def load_image(self, file_path):

        try:

            image = imageio.imread(file_path)

            metadata = {
                "name": file_path.split("\\")[-1]
            }

            return True, image, metadata, None

        except Exception as e:

            print("Error loading image:", e)

            return False, None, {}, None

    # -----------------------------
    # PROCESS IMAGE
    # -----------------------------

    def process_image(self, image_data, metadata, params):

        operation = params.get("operation")

        result = image_data.copy()

        if operation == "Gaussian Blur":

            result = filters.gaussian(
                result,
                sigma=2,
                preserve_range=True
            ).astype(np.uint8)

        elif operation == "Edge Detection":

            gray = color.rgb2gray(result)

            edges = filters.sobel(gray)

            result = (edges * 255).astype(np.uint8)

        return result
