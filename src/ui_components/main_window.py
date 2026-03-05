from PySide6.QtWidgets import *

"""(
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QComboBox, QSizePolicy, QLabel, QStackedWidget,

)"""
from PySide6.QtCore import Qt, Signal

from module_manager import ModuleManager
from image_data_store import ImageDataStore

import numpy as np

# Import napari viewer
import napari

class MainWindow(QMainWindow):
    def __init__(self, module_manager: ModuleManager):
        super().__init__()
        self.setWindowTitle("Modular Image Viewer (Python Native)")
        self.setGeometry(100, 100, 1200, 800)

        self.module_manager = module_manager
        self.image_data_store = ImageDataStore()

        self.setup_ui()
        self.connect_signals()
        self.module_manager.activate_module(self.module_selector.currentText()) # Activate default

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left Control Panel
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)

        self.load_button = QPushButton("Load Image...")
        left_layout.addWidget(self.load_button)

        left_layout.addWidget(QLabel("Select Module:"))
        self.module_selector = QComboBox()
        self.module_selector.addItems(self.module_manager.get_module_names())
        left_layout.addWidget(self.module_selector)

        self.module_control_stack = QStackedWidget()
        left_layout.addWidget(self.module_control_stack)
        left_layout.addStretch(1) # Push everything to top

        main_layout.addWidget(left_panel)

        # --- Central View Panel with Two Viewers ---
        # Create two separate napari viewers
        self.original_viewer = napari.Viewer(title="Original", show=False)
        self.processed_viewer = napari.Viewer(title="Processed", show=False)
        
        # Link the cameras so zoom/pan is synchronized.
        # The `camera.link()` method was removed in napari.
        # The new way is to connect camera events.
        self.original_viewer.camera.events.connect(self._sync_processed_viewer_camera)
        self.processed_viewer.camera.events.connect(self._sync_original_viewer_camera)

        # Get the Qt widgets for embedding
        original_widget = self.original_viewer.window.qt_viewer
        processed_widget = self.processed_viewer.window.qt_viewer

        # --- Create containers for viewers with titles ---
        # Original Viewer Container
        original_container = QWidget()
        original_layout = QVBoxLayout(original_container)
        original_layout.setContentsMargins(0, 5, 0, 0)
        original_layout.setSpacing(5)
        original_title = QLabel("Original Image")
        original_title.setAlignment(Qt.AlignCenter)
        original_layout.addWidget(original_title)
        original_layout.addWidget(original_widget)

        # Processed Viewer Container
        processed_container = QWidget()
        processed_layout = QVBoxLayout(processed_container)
        processed_layout.setContentsMargins(0, 5, 0, 0)
        processed_layout.setSpacing(5)
        processed_title = QLabel("Processed Image")
        processed_title.setAlignment(Qt.AlignCenter)
        processed_layout.addWidget(processed_title)
        processed_layout.addWidget(processed_widget)

        # Use a QSplitter to make the viewers resizable
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(original_container)
        splitter.addWidget(processed_container)
        
        main_layout.addWidget(splitter, 1) # Add splitter to main layout

    def connect_signals(self):
        self.load_button.clicked.connect(self.select_image_file)
        self.module_selector.currentTextChanged.connect(self.module_manager.activate_module)
        self.module_manager.module_activated.connect(self.update_control_panel)
        self.module_manager.image_loaded_and_processed.connect(self.update_napari_viewer)
        self.module_manager.clear_viewer_requested.connect(self.clear_viewer)

    def select_image_file(self):
        # Get supported formats from active module
        active_module = self.module_manager.get_active_module()
        if not active_module:
            QMessageBox.warning(self, "No Module", "Please select an image type module first.")
            return

        supported_exts = active_module.get_supported_formats()
        file_filter = f"Image Files ({' '.join(['*.' + ext for ext in supported_exts])});;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", file_filter)
        if file_path:
            self.module_manager.load_and_process_image(file_path)

    def update_control_panel(self, module_name: str, control_widget: QWidget):
        # Clear existing widgets in the stack and add the new one
        while self.module_control_stack.count() > 0:
            widget_to_remove = self.module_control_stack.widget(0)
            self.module_control_stack.removeWidget(widget_to_remove)
            widget_to_remove.deleteLater() # Ensure proper cleanup

        self.module_control_stack.addWidget(control_widget)
        self.module_control_stack.setCurrentWidget(control_widget)

        # Connect any specific signals from the control_widget to the manager for processing
        # Example: if control_widget has a 'process_button_clicked' signal
        # control_widget.process_button_clicked.connect(self.module_manager.apply_processing_to_current_image)
        # Or, the control_widget itself can call module_manager.apply_processing_to_current_image
        # which is often cleaner.

    def update_napari_viewer(self, image_data: np.ndarray, metadata: dict, session_id: str):
        """Routes image data to the correct viewer based on metadata."""
        layer_name = metadata.get('layer_name')
        
        viewer_to_update = None
        if layer_name == 'Original':
            viewer_to_update = self.original_viewer
            # When a new original image is loaded, clear the processed viewer
            # by hiding its layer.
            if 'Processed' in self.processed_viewer.layers:
                self.processed_viewer.layers['Processed'].visible = False
        elif layer_name == 'Processed':
            viewer_to_update = self.processed_viewer
        else:
            print(f"Warning: Unknown layer name '{layer_name}'. Not displaying.")
            return

        # To avoid KeyError on rapid layer removal/addition, update the layer
        # data if the layer exists, otherwise add a new layer.
        try:
            layer = viewer_to_update.layers[layer_name]
            layer.data = image_data
            layer.contrast_limits = metadata.get('contrast_limits', (image_data.min(), image_data.max()))
            layer.visible = True
        except KeyError:
            # Layer doesn't exist, so add it
            viewer_to_update.add_image(image_data, name=layer_name, contrast_limits=metadata.get('contrast_limits'))
            viewer_to_update.reset_view()

    def _sync_processed_viewer_camera(self, event):
        """Syncs the processed viewer camera to the original viewer's camera."""
        with self.processed_viewer.camera.events.blocker(self._sync_original_viewer_camera):
            self.processed_viewer.camera.center = self.original_viewer.camera.center
            self.processed_viewer.camera.zoom = self.original_viewer.camera.zoom
            self.processed_viewer.camera.angles = self.original_viewer.camera.angles

    def _sync_original_viewer_camera(self, event):
        """Syncs the original viewer camera to the processed viewer's camera."""
        with self.original_viewer.camera.events.blocker(self._sync_processed_viewer_camera):
            self.original_viewer.camera.center = self.processed_viewer.camera.center
            self.original_viewer.camera.zoom = self.processed_viewer.camera.zoom
            self.original_viewer.camera.angles = self.processed_viewer.camera.angles
        
    def clear_viewer(self):
        """Clears all layers from both napari viewers."""
        # Instead of clearing, just hide all layers. This is more stable.
        for viewer in [self.original_viewer, self.processed_viewer]:
            for layer in viewer.layers:
                layer.visible = False
        # No need to reset view, as the camera position is preserved
        # for the next loaded image.