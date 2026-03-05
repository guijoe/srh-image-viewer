from __future__ import annotations

import uuid

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from modules.i_image_module import IImageModule


class BasicModule(IImageModule):
    def get_name(self) -> str:
        return "Basic Image Processing"

    def get_supported_formats(self) -> list[str]:
        return ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]

    def create_control_widget(self, module_manager) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("Apply a simple brightness increase to the image."))

        process_btn = QPushButton("Increase Brightness (+50)")
        process_btn.clicked.connect(
            lambda: module_manager.apply_processing_to_current_image(
                {"brightness_delta": 50}
            )
        )
        layout.addWidget(process_btn)
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
                "contrast_limits": (float(np.min(image_data)), float(np.max(image_data))),
            }
            return True, image_data, metadata, str(uuid.uuid4())
        except Exception as exc:
            print(f"Failed to load image '{file_path}': {exc}")
            return False, None, {}, None

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict | None = None) -> np.ndarray:
        delta = int((params or {}).get("brightness_delta", 50))
        if np.issubdtype(image_data.dtype, np.integer):
            clipped = np.clip(
                image_data.astype(np.int32) + delta,
                0,
                np.iinfo(image_data.dtype).max,
            )
            return clipped.astype(image_data.dtype)
        return np.clip(image_data + delta, 0, 1 if np.max(image_data) <= 1 else 255)
