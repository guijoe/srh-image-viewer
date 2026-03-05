from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QComboBox, QStackedWidget, QDoubleSpinBox,
    QSpinBox, QGridLayout
)
from PySide6.QtCore import Signal
import numpy as np
import imageio
from scipy.ndimage import convolve, median_filter
from modules.i_image_module import IImageModule


# =========================================================
# BASE PARAMETER WIDGET
# =========================================================

class BaseParamsWidget(QWidget):
    def get_params(self):
        raise NotImplementedError


class NoParamsWidget(BaseParamsWidget):

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters required"))
        layout.addStretch()

    def get_params(self):
        return {}


# =========================================================
# PARAMETER WIDGETS
# =========================================================

class GaussianNoiseParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Mean"))
        self.mean = QDoubleSpinBox()
        self.mean.setValue(0)
        layout.addWidget(self.mean)

        layout.addWidget(QLabel("Standard Deviation"))
        self.std = QDoubleSpinBox()
        self.std.setValue(20)
        layout.addWidget(self.std)

    def get_params(self):
        return {
            "mean": self.mean.value(),
            "std": self.std.value()
        }


class SaltPepperParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Noise Probability"))

        self.prob = QDoubleSpinBox()
        self.prob.setSingleStep(0.01)
        self.prob.setValue(0.02)

        layout.addWidget(self.prob)

    def get_params(self):
        return {"prob": self.prob.value()}


class MedianParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Kernel Size"))

        self.kernel = QSpinBox()
        self.kernel.setValue(3)
        self.kernel.setSingleStep(2)

        layout.addWidget(self.kernel)

    def get_params(self):
        return {"kernel": self.kernel.value()}


class FourierParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Cutoff Radius"))

        self.radius = QSpinBox()
        self.radius.setValue(30)

        layout.addWidget(self.radius)

    def get_params(self):
        return {"radius": self.radius.value()}


class CannyParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Low Threshold"))

        self.low = QDoubleSpinBox()
        self.low.setValue(50)

        layout.addWidget(self.low)

        layout.addWidget(QLabel("High Threshold"))

        self.high = QDoubleSpinBox()
        self.high.setValue(150)

        layout.addWidget(self.high)

    def get_params(self):
        return {
            "low": self.low.value(),
            "high": self.high.value()
        }


class MorphologyParams(BaseParamsWidget):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Kernel Size"))

        self.kernel = QSpinBox()
        self.kernel.setValue(3)

        layout.addWidget(self.kernel)

    def get_params(self):
        return {"kernel": self.kernel.value()}


# =========================================================
# CONTROL PANEL
# =========================================================

class DydyControlsWidget(QWidget):

    process_requested = Signal(dict)

    def __init__(self, module_manager, parent=None):

        super().__init__(parent)

        self.module_manager = module_manager
        self.param_widgets = {}

        self.setup_ui()

    def setup_ui(self):

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("<h3>Control Panel</h3>"))

        self.operation_selector = QComboBox()

        layout.addWidget(self.operation_selector)

        self.params_stack = QStackedWidget()

        layout.addWidget(self.params_stack)

        operations = {
            "Gaussian Noise": GaussianNoiseParams,
            "Salt & Pepper Noise": SaltPepperParams,
            "Median Filter": MedianParams,
            "Fourier Low Pass Filter": FourierParams,
            "Histogram Equalization": NoParamsWidget,
            "Canny Edge Detector": CannyParams,
            "Erosion": MorphologyParams,
            "Dilation": MorphologyParams,
            "Opening": MorphologyParams,
            "Closing": MorphologyParams,
            "PSNR": NoParamsWidget
        }

        for name, widget_class in operations.items():

            widget = widget_class()

            self.params_stack.addWidget(widget)

            self.param_widgets[name] = widget

            self.operation_selector.addItem(name)

        self.apply_button = QPushButton("Apply Processing")

        layout.addWidget(self.apply_button)

        self.apply_button.clicked.connect(self._apply)

        self.operation_selector.currentTextChanged.connect(
            lambda name: self.params_stack.setCurrentWidget(self.param_widgets[name])
        )

    def _apply(self):

        op = self.operation_selector.currentText()

        params = self.param_widgets[op].get_params()

        params["operation"] = op

        self.process_requested.emit(params)


# =========================================================
# IMAGE MODULE
# =========================================================

class DydyImageModule(IImageModule):

    def __init__(self):

        super().__init__()

        self._controls_widget = None

    def get_name(self):
        return "Dydy Module"

    def get_supported_formats(self):

        return ["png", "jpg", "jpeg", "bmp"]

    def create_control_widget(self, parent=None, module_manager=None):

        if self._controls_widget is None:

            self._controls_widget = DydyControlsWidget(module_manager, parent)

            self._controls_widget.process_requested.connect(self._handle_request)

        return self._controls_widget

    def _handle_request(self, params):

        if self._controls_widget.module_manager:

            self._controls_widget.module_manager.apply_processing_to_current_image(params)

    def load_image(self, file_path):

        img = imageio.imread(file_path)

        metadata = {"name": file_path.split("/")[-1]}

        return True, img, metadata, None


    # =========================================================
    # MORPHOLOGY FUNCTIONS
    # =========================================================

    def erosion(self, image, k):

        padded = np.pad(image, k//2)

        windows = np.lib.stride_tricks.sliding_window_view(padded,(k,k))

        return np.min(windows, axis=(2,3))


    def dilation(self, image, k):

        padded = np.pad(image, k//2)

        windows = np.lib.stride_tricks.sliding_window_view(padded,(k,k))

        return np.max(windows, axis=(2,3))


    # =========================================================
    # MAIN PROCESSING
    # =========================================================

    def process_image(self, image_data, metadata, params):

        img = image_data.astype(float)

        operation = params["operation"]

        if img.ndim == 3:
            gray = np.mean(img[:,:,:3], axis=2)
        else:
            gray = img


        # ---------------- NOISE ----------------

        if operation == "Gaussian Noise":

            mean = params["mean"]
            std = params["std"]

            img = img + np.random.normal(mean, std, img.shape)


        elif operation == "Salt & Pepper Noise":

            prob = params["prob"]

            rnd = np.random.rand(*gray.shape)

            img[rnd < prob] = 0
            img[rnd > 1-prob] = 255


        # ---------------- MEDIAN ----------------

        elif operation == "Median Filter":

            k = params["kernel"]

            img = median_filter(gray, size=k)


        # ---------------- FOURIER ----------------

        elif operation == "Fourier Low Pass Filter":

            r = params["radius"]

            f = np.fft.fftshift(np.fft.fft2(gray))

            rows, cols = gray.shape

            mask = np.zeros_like(gray)

            mask[rows//2-r:rows//2+r, cols//2-r:cols//2+r] = 1

            f *= mask

            img = np.abs(np.fft.ifft2(np.fft.ifftshift(f)))


        # ---------------- HISTOGRAM ----------------

        elif operation == "Histogram Equalization":

            hist, bins = np.histogram(gray.flatten(),256,[0,256])

            cdf = hist.cumsum()

            cdf = 255 * cdf / cdf[-1]

            img = np.interp(gray.flatten(), bins[:-1], cdf).reshape(gray.shape)


        # ---------------- MORPHOLOGY ----------------

        elif operation == "Erosion":

            img = self.erosion(gray, params["kernel"])


        elif operation == "Dilation":

            img = self.dilation(gray, params["kernel"])


        elif operation == "Opening":

            img = self.dilation(self.erosion(gray, params["kernel"]), params["kernel"])


        elif operation == "Closing":

            img = self.erosion(self.dilation(gray, params["kernel"]), params["kernel"])


        # ---------------- CANNY ----------------

        elif operation == "Canny Edge Detector":

            low = params["low"]
            high = params["high"]

            Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
            Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])

            Ix = convolve(gray,Kx)
            Iy = convolve(gray,Ky)

            G = np.hypot(Ix,Iy)

            edges = np.zeros_like(G)

            edges[G > high] = 255

            edges[(G >= low) & (G <= high)] = 100

            img = edges


        # ---------------- PSNR ----------------

        elif operation == "PSNR":

            mse = np.mean((image_data - img)**2)

            psnr = 20 * np.log10(255/np.sqrt(mse))

            print("PSNR:", psnr)

            return image_data


        img = np.clip(img,0,255)

        # Ensure output has same number of dimensions as input to avoid napari layer issues
        if img.ndim != image_data.ndim:
            if image_data.ndim == 3 and img.ndim == 2:
                # Replicate grayscale result to 3 channels for RGB input
                img = np.stack([img] * 3, axis=2)
            elif image_data.ndim == 2 and img.ndim == 3:
                # Convert back to grayscale if input was 2D but output is 3D (unlikely in this module)
                img = np.mean(img[:, :, :3], axis=2)

        return img.astype(image_data.dtype)