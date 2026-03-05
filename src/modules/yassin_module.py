from __future__ import annotations

import uuid

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from modules.i_image_module import IImageModule


class YassinModule(IImageModule):

    def get_name(self) -> str:
        return "Yassin Image Processing"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

    def create_control_widget(self, module_manager) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(QLabel("Yassin Image Processing Tools"))

        # Brightness
        bright_btn = QPushButton("Yassin Boost (+50)")
        bright_btn.clicked.connect(
            lambda: module_manager.apply_processing_to_current_image(
                {"mode": "brightness"}
            )
        )
        layout.addWidget(bright_btn)

        # Gaussian Blur
        blur_btn = QPushButton("Gaussian Blur")
        blur_btn.clicked.connect(
            lambda: module_manager.apply_processing_to_current_image(
                {"mode": "gaussian"}
            )
        )
        layout.addWidget(blur_btn)

        # Histogram Equalization
        hist_btn = QPushButton("Histogram Equalization")
        hist_btn.clicked.connect(
            lambda: module_manager.apply_processing_to_current_image(
                {"mode": "histogram"}
            )
        )
        layout.addWidget(hist_btn)

        # Sobel Edge Detection
        sobel_btn = QPushButton("Sobel Edge Detection")
        sobel_btn.clicked.connect(
            lambda: module_manager.apply_processing_to_current_image(
                {"mode": "sobel"}
            )
        )
        layout.addWidget(sobel_btn)

        layout.addStretch(1)

        return widget

    def load_image(self, file_path: str) -> tuple[bool, np.ndarray | None, dict, str | None]:
        try:
            image = Image.open(file_path)
            image_data = np.array(image)

            metadata = {
                "source_path": file_path,
                "dtype": str(image_data.dtype),
                "shape": image_data.shape,
                "contrast_limits": (
                    float(np.min(image_data)),
                    float(np.max(image_data))
                ),
            }

            return True, image_data, metadata, str(uuid.uuid4())

        except Exception as exc:
            print(f"Failed to load image '{file_path}': {exc}")
            return False, None, {}, None

    def process_image(
        self,
        image_data: np.ndarray,
        metadata: dict,
        params: dict | None = None
    ) -> np.ndarray:

        mode = (params or {}).get("mode", "brightness")

        # ---------- Brightness ----------
        if mode == "brightness":
            delta = 50

            if np.issubdtype(image_data.dtype, np.integer):
                clipped = np.clip(
                    image_data.astype(np.int32) + delta,
                    0,
                    np.iinfo(image_data.dtype).max,
                )
                return clipped.astype(image_data.dtype)

            return np.clip(image_data + delta, 0, 255)

        # ---------- Gaussian Blur ----------
        if mode == "gaussian":

            kernel = np.array([
                [1,2,1],
                [2,4,2],
                [1,2,1]
            ]) / 16

            padded = np.pad(image_data, ((1,1),(1,1),(0,0)), mode="edge")
            output = np.zeros_like(image_data)

            for i in range(image_data.shape[0]):
                for j in range(image_data.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    output[i,j] = np.sum(region * kernel[:,:,None], axis=(0,1))

            return output.astype(image_data.dtype)

        # ---------- Histogram Equalization ----------
        if mode == "histogram":

            gray = np.mean(image_data, axis=2).astype(np.uint8)

            hist, _ = np.histogram(gray.flatten(), 256, [0,256])
            cdf = hist.cumsum()

            cdf = 255 * cdf / cdf[-1]

            result = cdf[gray]

            return np.stack([result]*3, axis=2).astype(np.uint8)

        # ---------- Sobel Edge Detection ----------
        if mode == "sobel":

            gray = np.mean(image_data, axis=2)

            sobel_x = np.array([
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
            ])

            sobel_y = np.array([
                [-1,-2,-1],
                [0,0,0],
                [1,2,1]
            ])

            padded = np.pad(gray, 1, mode="edge")

            gx = np.zeros_like(gray)
            gy = np.zeros_like(gray)

            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    region = padded[i:i+3, j:j+3]
                    gx[i,j] = np.sum(region * sobel_x)
                    gy[i,j] = np.sum(region * sobel_y)

            magnitude = np.sqrt(gx**2 + gy**2)

            magnitude = np.clip(magnitude, 0, 255)

            return np.stack([magnitude]*3, axis=2).astype(np.uint8)

        return image_data
