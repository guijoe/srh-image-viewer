import numpy as np
from abc import ABC, abstractmethod
from PySide6.QtWidgets import QWidget # Or PyQt6.QtWidgets

class IImageModule(ABC):
    @abstractmethod
    def get_name(self) -> str:
        """Returns the display name of the module (e.g., '2D Standard Images')."""
        pass

    @abstractmethod
    def get_supported_formats(self) -> list[str]:
        """Returns a list of file extensions or format identifiers this module handles."""
        pass

    @abstractmethod
    def create_control_widget(self, parent=None, module_manager=None) -> QWidget:
        """Creates and returns the PyQt/PySide widget for this module's controls."""
        pass

    @abstractmethod
    def load_image(self, file_path: str):
        """Loads an image file, stores it in ImageDataStore, and returns success status."""
        pass

    @abstractmethod
    def process_image(self, image_data: np.ndarray, metadata: dict) -> np.ndarray:
        """Applies processing to the image data and returns the new processed data."""
        pass

    # Optional: Add methods for events, e.g., on_image_loaded, on_active, on_deactivated