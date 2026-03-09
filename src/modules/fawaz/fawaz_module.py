import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton

from modules.i_image_module import IImageModule


class NoParamsWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No extra parameters are needed for this operation."))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}


class FawazControlsWidget(QWidget):
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Select Operation:"))

        self.operation_selector = QComboBox()
        self.operation_selector.addItems([
            "Grayscale (Luminance)",
            "Invert Colors",
            "Sepia",
            "Brighten",
            "Darken",
            "Threshold Black & White"
        ])
        layout.addWidget(self.operation_selector)

        self.params_widget = NoParamsWidget()
        layout.addWidget(self.params_widget)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self.apply_processing)
        layout.addWidget(self.apply_button)

        layout.addStretch()

    def apply_processing(self):
        params = self.params_widget.get_params()
        params["operation"] = self.operation_selector.currentText()
        self.module_manager.apply_processing_to_current_image(params)


class FawazImageModule(IImageModule):
    def get_name(self) -> str:
        return "Fawaz Module"

    def get_supported_formats(self) -> list:
        return ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

    def load_image(self, file_path: str):
        from skimage import io

        try:
            image_data = io.imread(file_path)
            image_data = np.array(image_data)

            metadata = {
                "name": file_path.split("/")[-1],
                "layer_name": "Original",
                "file_path": file_path
            }

            return True, image_data, metadata, None

        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def create_control_widget(self, module_manager) -> QWidget:
        return FawazControlsWidget(module_manager)

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get("operation")

        if processed_data.ndim == 3:
            img_float = processed_data.astype(np.float32)
        else:
            img_float = processed_data.astype(np.float32)

        if operation == "Grayscale (Luminance)":
            if processed_data.ndim == 3:
                gray = (
                    0.299 * img_float[:, :, 0] +
                    0.587 * img_float[:, :, 1] +
                    0.114 * img_float[:, :, 2]
                )
                gray = np.clip(gray, 0, 255).astype(image_data.dtype)
                processed_data = np.stack((gray, gray, gray), axis=-1)
            else:
                processed_data = processed_data.copy()

        elif operation == "Invert Colors":
            processed_data = 255 - img_float

        elif operation == "Sepia":
            if processed_data.ndim == 3:
                r = img_float[:, :, 0]
                g = img_float[:, :, 1]
                b = img_float[:, :, 2]

                sepia_r = 0.393 * r + 0.769 * g + 0.189 * b
                sepia_g = 0.349 * r + 0.686 * g + 0.168 * b
                sepia_b = 0.272 * r + 0.534 * g + 0.131 * b

                processed_data = np.stack((sepia_r, sepia_g, sepia_b), axis=-1)
            else:
                processed_data = processed_data.copy()

        elif operation == "Brighten":
            processed_data = img_float + 50

        elif operation == "Darken":
            processed_data = img_float - 50

        elif operation == "Threshold Black & White":
            if processed_data.ndim == 3:
                gray = (
                    0.299 * img_float[:, :, 0] +
                    0.587 * img_float[:, :, 1] +
                    0.114 * img_float[:, :, 2]
                )
            else:
                gray = img_float

            thresholded = np.where(gray >= 128, 255, 0).astype(image_data.dtype)

            if processed_data.ndim == 3:
                processed_data = np.stack((thresholded, thresholded, thresholded), axis=-1)
            else:
                processed_data = thresholded

        processed_data = np.clip(processed_data, 0, 255).astype(image_data.dtype)

        if metadata is None:
            metadata = {}

        p_min, p_max = processed_data.min(), processed_data.max()
        if p_min == p_max:
            metadata["contrast_limits"] = (float(p_min), float(p_max) + 0.0001)
        else:
            metadata["contrast_limits"] = (float(p_min), float(p_max))

        return processed_data # add this to run cd ~/Downloads/ImagingTechProject/Fawaz-agha/src && python main_app.py