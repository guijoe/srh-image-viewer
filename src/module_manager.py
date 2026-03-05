import os
import importlib
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QWidget
from modules.i_image_module import IImageModule
from image_data_store import ImageDataStore

class ModuleManager(QObject):
    module_activated = Signal(str, QWidget) # module_name, control_widget
    image_loaded_and_processed = Signal(object, dict, str) # image_data, metadata, session_id
    clear_viewer_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._modules: dict[str, IImageModule] = {}
        self._active_module: IImageModule = None
        self._load_modules_from_directory("modules") # Dynamically load

    def _load_modules_from_directory(self, path):
        print("loading modules from repository")
        # This is a simplified dynamic loading. For production, consider entry points.
        for item in os.listdir(path):
            print(item)
            module_path = os.path.join(path, item)
            if os.path.isdir(module_path) and not item.startswith('__'):
                try:
                    # Attempt to import module_name.module_name_file
                    # e.g., modules.two_d_images.two_d_module
                    module_name = item
                    print("Module name", module_name)
                    module_file_name = f"{item.replace('-', '_')}_module" # Convention: folder_name_module.py
                    
                    print(f"{path}.{module_name}.{module_file_name}")
                    module_spec = importlib.util.find_spec(f"{path}.{module_name}.{module_file_name}")

                    print("Module spec:", module_spec)
                    if module_spec:
                        module = importlib.util.module_from_spec(module_spec)
                        module_spec.loader.exec_module(module)

                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if isinstance(attr, type) and issubclass(attr, IImageModule) and attr is not IImageModule:
                                module_instance = attr()
                                self.register_module(module_instance)
                                print(f"Loaded module: {module_instance.get_name()}")
                                break # Found the module class
                except Exception as e:
                    print(f"Could not load module {item}: {e}")

    def register_module(self, module: IImageModule):
        if module.get_name() in self._modules:
            print(f"Warning: Module '{module.get_name()}' already registered. Overwriting.")
        self._modules[module.get_name()] = module

    def get_module_names(self) -> list[str]:
        return list(self._modules.keys())

    def get_module_by_name(self, name: str) -> IImageModule:
        return self._modules.get(name)

    def activate_module(self, module_name: str):
        module = self.get_module_by_name(module_name)
        if module:
            self._active_module = module
            control_widget = module.create_control_widget(module_manager=self)
            self.module_activated.emit(module_name, control_widget)
            print(f"Module '{module_name}' activated.")
        else:
            print(f"Module '{module_name}' not found.")

    def get_active_module(self) -> IImageModule:
        return self._active_module

    def load_and_process_image(self, file_path: str):
        if not self._active_module:
            print("No active module to load image.")
            return

        try:
            success, image_data, metadata, session_id = self._active_module.load_image(file_path)
            if success:
                self.clear_viewer_requested.emit() # Clear previous images

                # Store the loaded image data in the global store
                ImageDataStore().set_image(image_data, metadata, session_id)

                # Emit signal for the original image
                original_meta = metadata.copy()
                original_meta['layer_name'] = 'Original'
                self.image_loaded_and_processed.emit(image_data, original_meta, session_id)
            else:
                print(f"Failed to load image with active module {self._active_module.get_name()}")
        except Exception as e:
            print(f"Error loading image with module {self._active_module.get_name()}: {e}")

    def apply_processing_to_current_image(self, params: dict = None):
        if not self._active_module:
            print("No active module for processing.")
            return

        current_data, current_meta, current_session_id = ImageDataStore().get_image()
        if current_data is None:
            print("No image loaded to process.")
            return

        try:
            processed_data = self._active_module.process_image(current_data.copy(), current_meta.copy(), params)
            
            # Don't update the main store, just the viewer, to keep original data for next processing
            processed_meta = current_meta.copy()
            processed_meta['layer_name'] = 'Processed'
            self.image_loaded_and_processed.emit(processed_data, processed_meta, current_session_id)

            print("Image processed and viewer updated.")
        except Exception as e:
            print(f"Error processing image with module {self._active_module.get_name()}: {e}")