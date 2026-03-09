import numpy as np
from PIL import Image
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton
from PySide6.QtCore import Signal, Qt

from modules.i_image_module import IImageModule

class DanyControlsWidget(QWidget):
    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h3>Dany's Advanced Control Panel</h3>"))

        layout.addWidget(QLabel("Select Algorithm:"))
        self.operation_selector = QComboBox()
        self.operation_selector.addItems([
            "Original", 
            "Sobel Edge Detection", 
            "Unsharp Mask",
            "Canny Edge (Manual Multi-Stage)"
        ])
        layout.addWidget(self.operation_selector)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self._on_apply_clicked)
        layout.addWidget(self.apply_button)
        
        layout.addStretch()

    def _on_apply_clicked(self):
        params = {'operation': self.operation_selector.currentText()}
        self.process_requested.emit(params)

class danyModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Dany Advanced Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "webp", "tiff"]

    def create_control_widget(self, module_manager=None, parent=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = DanyControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            img = Image.open(file_path).convert("RGB")
            image_data = np.array(img)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        if image_data is None: return None

        operation = params.get('operation', 'Original')
        working_data = image_data.astype(np.float32)

        if operation == "Sobel Edge Detection":
            processed = self.apply_sobel(working_data)
        elif operation == "Unsharp Mask":
            processed = self.apply_unsharp(working_data)
        elif operation == "Canny Edge (Manual Multi-Stage)":
            processed = self.apply_canny(working_data)
        else:
            processed = working_data

        return processed.astype(np.uint8)

    def _convolve(self, img, kernel):
        h, w = img.shape[:2]
        padded = np.pad(img, 1, mode='edge')
        out = np.zeros_like(img)
        for i in range(3):
            for j in range(3):
                out += padded[i:i+h, j:j+w] * kernel[i, j]
        return out

    def apply_sobel(self, img):
        gray = 0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        gx = self._convolve(gray, kx)
        gy = self._convolve(gray, ky)
        mag = np.sqrt(gx**2 + gy**2)
        if mag.max() > 0: mag = (mag / mag.max()) * 255
        return np.stack([mag]*3, axis=-1)

    def apply_unsharp(self, img):
        blur_k = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        res = np.zeros_like(img)
        for c in range(3):
            blurred = self._convolve(img[..., c], blur_k)
            res[..., c] = img[..., c] + 1.5 * (img[..., c] - blurred)
        return np.clip(res, 0, 255)

    def apply_canny(self, img):
        """Manual Multi-stage Canny: Smoothing -> Gradient -> Non-Max Suppression -> Hysteresis"""
        # 1. Grayscale & Smooth
        gray = 0.2989 * img[...,0] + 0.5870 * img[...,1] + 0.1140 * img[...,2]
        gauss = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16.0
        smooth = self._convolve(gray, gauss)

        # 2. Gradients
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        gx = self._convolve(smooth, kx)
        gy = self._convolve(smooth, ky)
        mag = np.sqrt(gx**2 + gy**2)
        theta = np.arctan2(gy, gx) * 180 / np.pi
        theta[theta < 0] += 180

        # 3. Non-Maximum Suppression
        nms = np.zeros_like(mag)
        for i in range(1, mag.shape[0]-1):
            for j in range(1, mag.shape[1]-1):
                q, r = 255, 255
                angle = theta[i,j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                    q, r = mag[i, j+1], mag[i, j-1]
                elif (22.5 <= angle < 67.5):
                    q, r = mag[i+1, j-1], mag[i-1, j+1]
                elif (67.5 <= angle < 112.5):
                    q, r = mag[i+1, j], mag[i-1, j]
                elif (112.5 <= angle < 157.5):
                    q, r = mag[i-1, j-1], mag[i+1, j+1]
                if mag[i,j] >= q and mag[i,j] >= r: nms[i,j] = mag[i,j]

        # 4. Double Threshold/Hysteresis
        high = nms.max() * 0.2
        low = high * 0.5
        res = np.zeros_like(nms)
        res[nms >= high] = 255
        res[(nms >= low) & (nms < high)] = 50 
        return np.stack([res]*3, axis=-1)

Module = danyModule
