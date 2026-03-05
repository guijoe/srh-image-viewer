from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                             QComboBox, QStackedWidget, QDoubleSpinBox, QCheckBox)
from PySide6.QtCore import Signal, Qt
import numpy as np
import imageio

from modules.i_image_module import IImageModule

# --- Generic Parameter UI Components ---

class BaseParamsWidget(QWidget):
    # This signal tells the main UI that a slider/box has moved
    parametersChanged = Signal()
    
    def get_params(self) -> dict:
        raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None, text="Auto-calculated enhancement."):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        label = QLabel(text)
        label.setStyleSheet("font-style: italic; color: gray;")
        layout.addWidget(label)
        layout.addStretch()
    
    def get_params(self) -> dict: return {}

class DualParameterWidget(BaseParamsWidget):
    def __init__(self, label1, val1, min1, max1, label2, val2, min2, max2, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        
        # Parameter 1
        layout.addWidget(QLabel(label1))
        self.sb1 = QDoubleSpinBox()
        self.sb1.setRange(min1, max1)
        self.sb1.setValue(val1)
        self.sb1.setSingleStep(0.1)
        self.sb1.valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.sb1)
        
        # Parameter 2
        layout.addWidget(QLabel(label2))
        self.sb2 = QDoubleSpinBox()
        self.sb2.setRange(min2, max2)
        self.sb2.setValue(val2)
        self.sb2.setSingleStep(0.1)
        self.sb2.valueChanged.connect(lambda: self.parametersChanged.emit())
        layout.addWidget(self.sb2)
        
        layout.addStretch()

    def get_params(self) -> dict: 
        return {'p1': self.sb1.value(), 'p2': self.sb2.value()}

# --- Main Module Implementation ---

class SiddhImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str:
        return "Siddh Module"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "bmp", "tiff"]

    def load_image(self, file_path: str):
        try:
            image_data = imageio.imread(file_path)
            metadata = {'name': file_path.split('/')[-1]}
            return True, image_data, metadata, None
        except Exception as e:
            return False, None, {}, str(e)

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = SiddhControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            # Passes the request to the main framework to update the display
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    # --- NumPy Image Processing Algorithms ---

    def _ensure_rgb(self, img):
        """Converts grayscale (H,W) to (H,W,3) if necessary."""
        if img.ndim == 2:
            return np.stack([img, img, img], axis=-1)
        return img

    def _apply_sharpen(self, img, strength):
        laplacian = np.zeros_like(img)
        # Using a simple 4-neighbor cross kernel
        laplacian[1:-1, 1:-1] = (4 * img[1:-1, 1:-1] - img[:-2, 1:-1] - 
                                 img[2:, 1:-1] - img[1:-1, :-2] - img[1:-1, 2:])
        return img + (laplacian * strength)

    def _apply_linear_stretch(self, img, new_min=0, new_max=255):
        c_min, c_max = np.min(img), np.max(img)
        if c_max == c_min: return img
        return (img - c_min) * ((new_max - new_min) / (c_max - c_min)) + new_min

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        img = image_data.copy().astype(float)
        op = params.get('operation')

        # 1-4: Contrast & Lighting
        if op == "Contrast Stretching":
            img = self._apply_linear_stretch(img, params.get('p1', 0), params.get('p2', 255))
        elif op == "Brightness & Contrast":
            img = params.get('p2', 1.0) * (img - 128) + 128 + params.get('p1', 0)
        elif op == "Gamma Correction":
            img = 255 * (np.clip(img / 255, 0, 1) ** params.get('p1', 1.0))
        elif op == "Log Transformation":
            img = (255 / np.log(1 + np.max(img) + 1e-5)) * np.log(1 + img)

        # 5-8: Spatial Filters
        elif op == "Edge Sharpening":
            img = self._apply_sharpen(img, params.get('p1', 1.0))
        elif op == "Mean Blur":
            res = np.zeros_like(img)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    res += np.roll(np.roll(img, i, axis=0), j, axis=1)
            img = res / 9
        elif op == "Solarize Effect":
            threshold = params.get('p1', 128)
            img = np.where(img < threshold, img, 255 - img)
        elif op == "Color Quantization":
            levels = max(2, int(params.get('p1', 4)))
            img = np.floor(img / (256/levels)) * (256/levels)

        # 9-13: Color & Channel Operations
        elif op == "Sepia Filter":
            img = self._ensure_rgb(img)
            r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
            img[:,:,0] = (r * .393) + (g *.769) + (b * .189)
            img[:,:,1] = (r * .349) + (g *.686) + (b * .168)
            img[:,:,2] = (r * .272) + (g *.534) + (b * .131)
        elif op == "Intensity Threshold":
            img = np.where(img > params.get('p1', 128), 255, 0)
        elif op == "Negative/Inversion":
            img = 255 - img
        elif op == "Channel Balance":
            img = self._ensure_rgb(img)
            img[:,:,0] *= params.get('p1', 1.0) # Red
            img[:,:,2] *= params.get('p2', 1.0) # Blue
        elif op == "Grayscale conversion":
            img = self._ensure_rgb(img)
            gray = 0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]
            img = np.stack([gray, gray, gray], axis=-1)

        # 14-15: Complex Pipelines
        elif op == "Cinematic Pipeline":
            img = 255 * (np.clip(img / 255, 0, 1) ** 1.1) 
            img = self._apply_linear_stretch(img, 0, 255)
            img = self._apply_sharpen(img, 0.7)
        elif op == "HDR Pipeline":
            img = (255 / np.log(1 + np.max(img) + 1e-5)) * np.log(1 + img)
            img = self._apply_linear_stretch(img, 0, 255)
            img = self._apply_sharpen(img, 0.4)

        return np.clip(img, 0, 255).astype(image_data.dtype)

# --- UI Interface Class ---

class SiddhControlsWidget(QWidget):
    process_requested = Signal(dict)
    
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.widgets = {}
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Siddh Suite</h2>"))
        
        # Live Preview Toggle
        self.live_preview = QCheckBox("Live Preview")
        self.live_preview.setChecked(True)
        layout.addWidget(self.live_preview)
        
        self.selector = QComboBox()
        layout.addWidget(self.selector)
        
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        # Configuration mapping
        config = {
            "Cinematic Pipeline": NoParamsWidget,
            "HDR Pipeline": NoParamsWidget,
            "Contrast Stretching": lambda: DualParameterWidget("Min Out", 0, 0, 255, "Max Out", 255, 0, 255),
            "Edge Sharpening": lambda: DualParameterWidget("Strength", 1.0, 0, 5, "Unused", 0, 0, 0),
            "Brightness & Contrast": lambda: DualParameterWidget("Brightness", 0, -255, 255, "Contrast", 1.2, 0.1, 4.0),
            "Gamma Correction": lambda: DualParameterWidget("Gamma", 0.8, 0.1, 3.0, "Unused", 0, 0, 0),
            "Color Quantization": lambda: DualParameterWidget("Levels (2-16)", 4, 2, 16, "Unused", 0, 0, 0),
            "Solarize Effect": lambda: DualParameterWidget("Threshold", 128, 0, 255, "Unused", 0, 0, 0),
            "Intensity Threshold": lambda: DualParameterWidget("Level", 128, 0, 255, "Unused", 0, 0, 0),
            "Channel Balance": lambda: DualParameterWidget("Red Gain", 1.0, 0, 2, "Blue Gain", 1.0, 0, 2),
            "Log Transformation": NoParamsWidget,
            "Mean Blur": NoParamsWidget,
            "Sepia Filter": NoParamsWidget,
            "Negative/Inversion": NoParamsWidget,
            "Grayscale conversion": NoParamsWidget,
        }

        for name, factory in config.items():
            w = factory()
            # Connect the signal from the parameter widget to our local handler
            if hasattr(w, 'parametersChanged'):
                w.parametersChanged.connect(self._on_param_changed)
            
            self.stack.addWidget(w)
            self.widgets[name] = w
            self.selector.addItem(name)

        self.run_btn = QPushButton("Process Image")
        self.run_btn.setStyleSheet("background: #0984e3; color: white; font-weight: bold; height: 35px; border-radius: 5px;")
        self.run_btn.clicked.connect(self._run)
        layout.addWidget(self.run_btn)
        
        # Connect selector to stack and trigger run if live
        self.selector.currentTextChanged.connect(self._on_op_changed)

    def _on_op_changed(self, name):
        self.stack.setCurrentWidget(self.widgets[name])
        if self.live_preview.isChecked():
            self._run()

    def _on_param_changed(self):
        if self.live_preview.isChecked():
            self._run()

    def _run(self):
        name = self.selector.currentText()
        params = self.widgets[name].get_params()
        params['operation'] = name
        self.process_requested.emit(params)