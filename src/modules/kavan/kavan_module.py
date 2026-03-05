import time
import numpy as np
import imageio.v3 as iio
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QComboBox, QStackedWidget, QDoubleSpinBox, QSpinBox, 
                             QGridLayout, QProgressBar, QApplication)
from PySide6.QtCore import Qt, Signal
from numpy.lib.stride_tricks import sliding_window_view

from modules.i_image_module import IImageModule

def _get_gray(image):
    if image.ndim == 3:
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])
    return image.astype(float)

def _convolve_np(image, kernel):
    hk, wk = kernel.shape
    pad_h, pad_w = hk // 2, wk // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    windows = sliding_window_view(padded, (hk, wk))
    return np.einsum('ij,klij->kl', kernel, windows)


class BaseParamsWidget(QWidget):
    def get_params(self) -> dict: raise NotImplementedError

class NoParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters needed. Click Apply!"))
    def get_params(self) -> dict: return {}

class GammaParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Gamma (0.1 - 5.0):"))
        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.1, 5.0); self.spin.setValue(1.0)
        layout.addWidget(self.spin)
    def get_params(self) -> dict: return {'gamma': self.spin.value()}

class MorphParamsWidget(BaseParamsWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.op = QComboBox(); self.op.addItems(["Erosion", "Dilation"])
        layout.addWidget(QLabel("Operation:")); layout.addWidget(self.op)
        self.rad = QSpinBox(); self.rad.setRange(1, 10); self.rad.setValue(3)
        layout.addWidget(QLabel("Radius:")); layout.addWidget(self.rad)
    def get_params(self) -> dict: return {'morph_op': self.op.currentText(), 'radius': self.rad.value()}


class KavanControlsWidget(QWidget):
    process_requested = Signal(dict)
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.param_widgets = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        header = QLabel("<h3>Kavan Control Panel</h3>")
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)
        
        self.op_selector = QComboBox()
        self.stack = QStackedWidget()
        
        self.ops_config = {
            "Image Negative": NoParamsWidget,
            "Log Transformation": NoParamsWidget,
            "Power Law (Gamma)": GammaParamsWidget,
            "Morphological Ops": MorphParamsWidget,
            "Sobel Edge Detect": NoParamsWidget,
            "Vampire Mode": NoParamsWidget,
            "Rainbow Shift": NoParamsWidget  # <--- Added to UI
        }

        for name, widget_cls in self.ops_config.items():
            w = widget_cls()
            self.stack.addWidget(w)
            self.param_widgets[name] = w
            self.op_selector.addItem(name)

        self.op_selector.currentTextChanged.connect(
            lambda name: self.stack.setCurrentWidget(self.param_widgets[name])
        )

        self.apply_btn = QPushButton("Apply Processing")
        self.apply_btn.clicked.connect(self._on_apply_clicked)
        
        layout.addWidget(self.op_selector)
        layout.addWidget(self.stack)
        layout.addWidget(self.apply_btn)

    def _on_apply_clicked(self):
        name = self.op_selector.currentText()
        params = self.param_widgets[name].get_params()
        params['operation'] = name
        self.process_requested.emit(params)

class KavanImageModule(IImageModule):
    def __init__(self):
        super().__init__()
        self._controls_widget = None

    def get_name(self) -> str: return "Kavan Module"
    def get_supported_formats(self) -> list[str]: return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        if self._controls_widget is None:
            self._controls_widget = KavanControlsWidget(module_manager, parent)
            self._controls_widget.process_requested.connect(self._handle_processing_request)
        return self._controls_widget

    def _handle_processing_request(self, params: dict):
        if self._controls_widget and self._controls_widget.module_manager:
            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path: str):
        try:
            image_data = iio.imread(file_path)
            if image_data.ndim == 2:
                image_data = np.stack([image_data]*3, axis=-1)
            elif image_data.shape[2] == 4:
                image_data = image_data[..., :3]
            return True, image_data, {'name': file_path}, str(id(image_data))
        except Exception:
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        start_time = time.perf_counter()
        op = params.get('operation')
        img = image_data.astype(np.float32)
        is_rgb = img.ndim == 3
        res = img

        if op == "Image Negative":
            res = 255 - img

        elif op == "Log Transformation":
            c = 255 / np.log(1 + np.max(img)) if np.max(img) > 0 else 1
            res = c * np.log(1 + img)

        elif op == "Power Law (Gamma)":
            gamma = params.get('gamma', 1.0)
            res = 255 * (img / 255)**gamma

        elif op == "Vampire Mode":
            gray = _get_gray(img)
            h, w = gray.shape[:2]
            vamp_res = np.zeros((h, w, 3), dtype=np.float32)
            vamp_res[..., 0] = np.clip(gray * 1.3, 0, 255) 
            res = vamp_res

        elif op == "Rainbow Shift" and is_rgb: 
            res = np.roll(img, shift=1, axis=-1)

        elif op == "Morphological Ops":
            gray = _get_gray(img)
            binary = (gray > np.mean(gray)).astype(np.float32)
            r = params.get('radius', 3)
            size = r * 2 + 1
            pad = np.pad(binary, r, mode='constant')
            win = sliding_window_view(pad, (size, size))
            b_res = np.amax(win, axis=(2, 3)) if params.get('morph_op') == "Dilation" else np.amin(win, axis=(2, 3))
            res = np.stack([b_res * 255]*3, axis=-1) if is_rgb else b_res * 255

        elif op == "Sobel Edge Detect":
            gray = _get_gray(img)
            kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            ky = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
            gx, gy = _convolve_np(gray, kx), _convolve_np(gray, ky)
            edge = np.sqrt(gx**2 + gy**2)
            res = np.stack([edge]*3, axis=-1) if is_rgb else edge

        print(f"[Kavan] {op} executed in {time.perf_counter()-start_time:.4f}s")
        return np.clip(res, 0, 255).astype(np.uint8)