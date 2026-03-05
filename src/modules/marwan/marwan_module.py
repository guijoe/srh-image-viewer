import numpy as np
import imageio
from scipy.ndimage import gaussian_filter, sobel
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QDoubleSpinBox, QComboBox, QPushButton
from modules.i_image_module import IImageModule

class MarwanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        # This name shows up in the app dropdown
        return "Marwan's Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

    def load_image(self, file_path: str):
        """Standard image loading required by the project framework."""
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            print(f"Error loading image: {e}")
            return False, None, {}, None

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        """Creates the UI panel with all your operations and sliders."""
        self._controls_widget = QWidget(parent)
        layout = QVBoxLayout(self._controls_widget)

        layout.addWidget(QLabel("<h3>Marwan's Panel</h3>"))
        
        # Dropdown for selecting the effect
        self.op_selector = QComboBox()
        self.op_selector.addItems([
            "Gaussian Blur", 
            "Contrast Stretching", 
            "Invert Colors", 
            "Edge Detection (Sobel)"
        ])
        layout.addWidget(QLabel("Select Operation:"))
        layout.addWidget(self.op_selector)

        # Sigma Control (Used for Blur)
        layout.addWidget(QLabel("Blur Sigma (Intensity):"))
        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.1, 20.0)
        self.sigma_spin.setValue(1.5)
        layout.addWidget(self.sigma_spin)

        # Max Control (Used for Contrast)
        layout.addWidget(QLabel("Contrast Max (0-255):"))
        self.max_spin = QDoubleSpinBox()
        self.max_spin.setRange(0, 255)
        self.max_spin.setValue(255)
        layout.addWidget(self.max_spin)

        # Apply Button
        apply_btn = QPushButton("Apply Effect")
        layout.addWidget(apply_btn)

        # Sends the data to the process_image function
        apply_btn.clicked.connect(lambda: module_manager.apply_processing_to_current_image({
            'operation': self.op_selector.currentText(),
            'sigma': self.sigma_spin.value(),
            'new_max': self.max_spin.value()
        }))

        return self._controls_widget

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        """The mathematical logic for each image effect."""
        operation = params.get('operation')
        
        # 1. INVERT COLORS (Point Operation)
        if operation == "Invert Colors":
            # Math: 255 minus the pixel value
            return (255 - image_data).astype(image_data.dtype)

        # 2. GAUSSIAN BLUR (Spatial Filtering)
        elif operation == "Gaussian Blur":
            sigma = params.get('sigma', 1.0)
            if image_data.ndim == 3: # Color image
                return gaussian_filter(image_data, sigma=(sigma, sigma, 0)).astype(image_data.dtype)
            return gaussian_filter(image_data, sigma=sigma).astype(image_data.dtype)

        # 3. CONTRAST STRETCHING (Linear Scaling)
        elif operation == "Contrast Stretching":
            img_float = image_data.astype(float)
            curr_min, curr_max = np.min(img_float), np.max(img_float)
            new_max = params.get('new_max', 255.0)
            if curr_max == curr_min: return image_data
            processed = (img_float - curr_min) * (new_max / (curr_max - curr_min))
            return np.clip(processed, 0, 255).astype(image_data.dtype)

        # 4. SOBEL EDGE DETECTION (Gradient Calculation)
        elif operation == "Edge Detection (Sobel)":
            img_float = image_data.astype(float)
            if img_float.ndim == 3: # Handle RGB
                dx = sobel(img_float, axis=0)
                dy = sobel(img_float, axis=1)
            else: # Handle Grayscale
                dx = sobel(img_float, axis=0)
                dy = sobel(img_float, axis=1)
            
            # Combine horizontal and vertical gradients
            magnitude = np.hypot(dx, dy)
            # Scale to 0-255
            max_val = np.max(magnitude)
            if max_val != 0:
                magnitude *= 255.0 / max_val
            return magnitude.astype(image_data.dtype)

        return image_data