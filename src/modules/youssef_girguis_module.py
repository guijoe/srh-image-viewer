from __future__ import annotations

import uuid
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QDoubleSpinBox, QPushButton
)

from modules.i_image_module import IImageModule


def _read_image(file_path: str) -> np.ndarray:
    """Read image file into numpy array."""
    try:
        import imageio.v3 as iio
        return np.asarray(iio.imread(file_path))
    except Exception:
        from PIL import Image
        return np.asarray(Image.open(file_path))


def _dtype_maxv(img: np.ndarray) -> float:
    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    img_f = img.astype(np.float32)
    m = float(np.max(img_f)) if img_f.size else 1.0
    return m if m > 1.5 else 1.0


def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32, copy=False)


def _split_alpha(img_f: np.ndarray):
    if img_f.ndim == 3 and img_f.shape[2] == 4:
        return img_f[..., :3], img_f[..., 3:4]
    return img_f, None


def _recombine_alpha(rgb_f: np.ndarray, alpha):
    if alpha is None:
        return rgb_f
    return np.concatenate([rgb_f, alpha], axis=2)


def _to_grayscale_float(img_f: np.ndarray) -> np.ndarray:
    if img_f.ndim == 2:
        return img_f
    if img_f.ndim == 3:
        rgb, _ = _split_alpha(img_f)
        if rgb.shape[2] >= 3:
            r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
            return 0.299 * r + 0.587 * g + 0.114 * b
        return np.mean(rgb, axis=2)
    return img_f


def _as_same_channels_from_gray(gray_f: np.ndarray, like_img_f: np.ndarray) -> np.ndarray:
    if like_img_f.ndim == 2:
        return gray_f
    if like_img_f.ndim == 3:
        rgb_like, alpha = _split_alpha(like_img_f)
        g3 = np.repeat(gray_f[..., None], rgb_like.shape[2], axis=2)
        return _recombine_alpha(g3, alpha)
    return gray_f


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.05)
    radius = int(max(1, round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _convolve_1d_reflect(arr: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    pad = kernel.size // 2
    pad_width = [(0, 0)] * arr.ndim
    pad_width[axis] = (pad, pad)
    padded = np.pad(arr, pad_width=pad_width, mode="reflect")

    def conv_line(x):
        return np.convolve(x, kernel, mode="valid").astype(np.float32)

    return np.apply_along_axis(conv_line, axis, padded)


def _gaussian_blur(img_f: np.ndarray, sigma: float) -> np.ndarray:
    k = _gaussian_kernel_1d(sigma)
    out = _convolve_1d_reflect(img_f, k, axis=0)
    out = _convolve_1d_reflect(out, k, axis=1)
    return out


def _sobel_edges(gray_f: np.ndarray) -> np.ndarray:
    gx_k = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    gy_k = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)

    def conv2(img, k):
        p = 1
        padded = np.pad(img, ((p,p),(p,p)), mode="reflect")
        h, w = img.shape
        out = np.zeros((h,w), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                out += k[i,j] * padded[i:i+h, j:j+w]
        return out

    gx = conv2(gray_f, gx_k)
    gy = conv2(gray_f, gy_k)
    mag = np.sqrt(gx*gx + gy*gy)
    return mag.astype(np.float32)


class YoussefGirguisControlsWidget(QWidget):

    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("Operation:"))

        self.operation = QComboBox()
        self.operation.addItems([
            "Grayscale",
            "Threshold (Binary)",
            "Gaussian Blur",
            "Sobel Edge",
            "Sharpen (Unsharp Mask)",
        ])
        self.operation.currentTextChanged.connect(self._on_op_change)

        row.addWidget(self.operation)
        layout.addLayout(row)

        self.thr_label = QLabel("Threshold (0-255):")
        self.thr = QDoubleSpinBox()
        self.thr.setRange(0,255)
        self.thr.setValue(128)

        self.sigma_label = QLabel("Sigma (blur strength):")
        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.05,10)
        self.sigma.setValue(1.0)

        self.amount_label = QLabel("Sharpen amount:")
        self.amount = QDoubleSpinBox()
        self.amount.setRange(0,5)
        self.amount.setValue(1.0)

        layout.addWidget(self.thr_label)
        layout.addWidget(self.thr)
        layout.addWidget(self.sigma_label)
        layout.addWidget(self.sigma)
        layout.addWidget(self.amount_label)
        layout.addWidget(self.amount)

        btn = QPushButton("Apply Processing")
        btn.clicked.connect(self._apply)
        layout.addWidget(btn)

        layout.addStretch()

        self._on_op_change(self.operation.currentText())

    def _on_op_change(self, op):
        self.thr_label.setVisible(op=="Threshold (Binary)")
        self.thr.setVisible(op=="Threshold (Binary)")
        self.sigma_label.setVisible(op in ["Gaussian Blur","Sharpen (Unsharp Mask)"])
        self.sigma.setVisible(op in ["Gaussian Blur","Sharpen (Unsharp Mask)"])
        self.amount_label.setVisible(op=="Sharpen (Unsharp Mask)")
        self.amount.setVisible(op=="Sharpen (Unsharp Mask)")

    def _apply(self):
        op = self.operation.currentText()
        params={"operation":op}

        if op=="Threshold (Binary)":
            params["threshold"]=float(self.thr.value())

        elif op=="Gaussian Blur":
            params["sigma"]=float(self.sigma.value())

        elif op=="Sharpen (Unsharp Mask)":
            params["sigma"]=float(self.sigma.value())
            params["amount"]=float(self.amount.value())

        self.module_manager.apply_processing_to_current_image(params)


class YoussefGirguisImageModule(IImageModule):

    def get_name(self):
        return "Youssef Girguis Module"

    def get_supported_formats(self):
        return [".png",".jpg",".jpeg",".bmp",".tif",".tiff"]

    def create_control_widget(self,module_manager):
        return YoussefGirguisControlsWidget(module_manager)

    def load_image(self,file_path):

        try:
            image_data=_read_image(file_path)

            metadata={
                "source_path":file_path,
                "dtype":str(image_data.dtype),
                "shape":tuple(image_data.shape),
            }

            session_id=str(uuid.uuid4())

            return True,image_data,metadata,session_id

        except Exception as e:
            print(e)
            return False,None,{}, ""

    def process_image(self,image_data,metadata,params):

        if params is None:
            return image_data

        op=params.get("operation","")
        img=image_data
        img_f=_to_float(img)
        maxv=_dtype_maxv(img)

        if op=="Grayscale":
            gray=_to_grayscale_float(img_f)
            out_f=_as_same_channels_from_gray(gray,img_f)

        elif op=="Threshold (Binary)":
            thr=params.get("threshold",128)
            gray=_to_grayscale_float(img_f)
            bin_f=np.where(gray>=thr,maxv,0)
            out_f=_as_same_channels_from_gray(bin_f,img_f)

        elif op=="Gaussian Blur":
            sigma=params.get("sigma",1)
            out_f=_gaussian_blur(img_f,sigma)

        elif op=="Sobel Edge":
            gray=_to_grayscale_float(img_f)
            mag=_sobel_edges(gray)
            out_f=_as_same_channels_from_gray(mag,img_f)

        elif op=="Sharpen (Unsharp Mask)":
            sigma=params.get("sigma",1)
            amount=params.get("amount",1)
            blurred=_gaussian_blur(img_f,sigma)
            out_f=img_f+amount*(img_f-blurred)

        else:
            out_f=img_f

        if np.issubdtype(img.dtype,np.integer):
            info=np.iinfo(img.dtype)
            out_f=np.clip(out_f,info.min,info.max)
            return out_f.astype(img.dtype)

        return out_f.astype(img.dtype)


ImageModule = YoussefGirguisImageModule
