from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from PySide6.QtWidgets import QWidget


class IImageModule(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        pass

    @abstractmethod
    def create_control_widget(self, module_manager) -> QWidget:
        pass

    @abstractmethod
    def load_image(
        self, file_path: str
    ) -> tuple[bool, Optional[np.ndarray], dict, Optional[str]]:
        pass

    @abstractmethod
    def process_image(
        self, image_data: np.ndarray, metadata: dict, params: Optional[dict] = None
    ) -> np.ndarray:
        pass
