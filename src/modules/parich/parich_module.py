import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSpinBox, QComboBox, QPushButton

# Assuming your project template uses a generic interface for modules
from modules.i_image_module import IImageModule

# --- UI WIDGETS FOR PARAMETERS ---

class NoParamsWidget(QWidget):
    """A generic widget for operations that don't require user input."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("No parameters needed for this operation."))
        layout.addStretch()

    def get_params(self) -> dict:
        return {}

class KuwaharaParamsWidget(QWidget):
    """Widget to control the Kuwahara filter radius."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Brush Size / Radius (1 to 10):"))
        self.radius_spinbox = QSpinBox()
        self.radius_spinbox.setRange(1, 10)
        self.radius_spinbox.setValue(3)
        layout.addWidget(self.radius_spinbox)
        layout.addStretch()

    def get_params(self) -> dict:
        return {'radius': self.radius_spinbox.value()}

# --- CONTROL PANEL ---

class ParichControlsWidget(QWidget):
    def __init__(self, module_manager, parent=None):
        super().__init__(parent)
        self.module_manager = module_manager
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Dictionary of operations mapping to their specific UI widgets
        self.operations = {
            "1. Grayscale (Luminance)": NoParamsWidget,
            "2. Color Inversion (Negative)": NoParamsWidget,
            "3. Sepia Tone": NoParamsWidget,
            "4. Sobel Edge Detection": NoParamsWidget,
            "5. Kuwahara (Watercolor)": KuwaharaParamsWidget,
            "6. AI Saliency (Eye Tracking)": NoParamsWidget,
        }
        
        layout.addWidget(QLabel("Select Operation:"))
        self.operation_selector = QComboBox()
        self.operation_selector.addItems(list(self.operations.keys()))
        self.operation_selector.currentTextChanged.connect(self.on_operation_changed)
        layout.addWidget(self.operation_selector)

        self.params_container = QVBoxLayout()
        layout.addLayout(self.params_container)

        self.apply_button = QPushButton("Apply Processing")
        self.apply_button.clicked.connect(self.apply_processing)
        layout.addWidget(self.apply_button)
        
        layout.addStretch()
        self.current_params_widget = None
        self.on_operation_changed(self.operation_selector.currentText())

    def on_operation_changed(self, operation_name):
        if self.current_params_widget:
            self.current_params_widget.deleteLater()
            
        widget_class = self.operations.get(operation_name, NoParamsWidget)
        self.current_params_widget = widget_class()
        self.params_container.addWidget(self.current_params_widget)

    def apply_processing(self):
        operation = self.operation_selector.currentText()
        params = self.current_params_widget.get_params()
        params['operation'] = operation
        self.module_manager.apply_processing_to_current_image(params)

# --- MAIN MODULE CLASS ---

class ParichImageModule(IImageModule):
    def get_name(self) -> str:
        return "Parich Module"

    def get_supported_formats(self) -> list:
        return ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]

    def load_image(self, file_path: str):
        """Loads an image from the given file path."""
        from skimage import io
        import numpy as np
        
        try:
            # Read the image using scikit-image
            image_data = io.imread(file_path)
            image_data = np.array(image_data)
            
            # Create the metadata dictionary expected by the template
            metadata = {
                'name': file_path.split('/')[-1],
                'layer_name': 'Original',
                'file_path': file_path
            }
            
            # Return exactly the 4 items the template expects:
            # Success (Bool), Image Data (Array), Metadata (Dict), Session ID (None)
            return True, image_data, metadata, None
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return False, None, {}, None

    def create_control_widget(self, module_manager) -> QWidget:
        return ParichControlsWidget(module_manager)

    def process_image(self, image_data: np.ndarray, metadata: dict, params: dict) -> np.ndarray:
        processed_data = image_data.copy()
        operation = params.get('operation')
        img_float = processed_data.astype(np.float32)

        # ---------------------------------------------------------
        # 1. GRAYSCALE CONVERSION
        # ---------------------------------------------------------
        if operation == "1. Grayscale (Luminance)":
            if img_float.ndim == 3:
                gray = np.dot(img_float[..., :3], [0.299, 0.587, 0.114])
                processed_data = np.stack((gray,)*3, axis=-1)
            
        # ---------------------------------------------------------
        # 2. COLOR INVERSION
        # ---------------------------------------------------------
        elif operation == "2. Color Inversion (Negative)":
            processed_data = 255.0 - img_float

        # ---------------------------------------------------------
        # 3. SEPIA TONE
        # ---------------------------------------------------------
        elif operation == "3. Sepia Tone":
            if img_float.ndim == 2:
                img_color = np.stack((img_float,)*3, axis=-1)
            else:
                img_color = img_float[..., :3]

            sepia_matrix = np.array([
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131]
            ])
            
            sepia_img = np.dot(img_color, sepia_matrix.T)
            processed_data = sepia_img

        # ---------------------------------------------------------
        # 4. SOBEL EDGE DETECTION
        # ---------------------------------------------------------
        elif operation == "4. Sobel Edge Detection":
            if img_float.ndim == 3:
                gray = np.dot(img_float[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = img_float

            pad_img = np.pad(gray, 1, mode='edge')

            gx = (pad_img[:-2, 2:] - pad_img[:-2, :-2]) + \
                 2 * (pad_img[1:-1, 2:] - pad_img[1:-1, :-2]) + \
                 (pad_img[2:, 2:] - pad_img[2:, :-2])
                 
            gy = (pad_img[2:, :-2] - pad_img[:-2, :-2]) + \
                 2 * (pad_img[2:, 1:-1] - pad_img[:-2, 1:-1]) + \
                 (pad_img[2:, 2:] - pad_img[:-2, 2:])

            magnitude = np.sqrt(gx**2 + gy**2)
            if magnitude.max() > 0:
                magnitude = (magnitude / magnitude.max()) * 255.0
            
            processed_data = np.stack((magnitude,)*3, axis=-1) if image_data.ndim == 3 else magnitude

        # ---------------------------------------------------------
        # 5. KUWAHARA (WATERCOLOR) FILTER
        # ---------------------------------------------------------
        elif operation == "5. Kuwahara (Watercolor)":
            radius = params.get('radius', 3)
            k = radius + 1 
            
            if img_float.ndim == 3:
                gray = np.dot(img_float[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = img_float
                
            pad_gray = np.pad(gray, radius, mode='reflect')
            pad_image = np.pad(img_float, ((radius, radius), (radius, radius), (0, 0))) if img_float.ndim == 3 else np.pad(img_float, radius, mode='reflect')
            
            H, W = gray.shape
            view_gray = sliding_window_view(pad_gray, (k, k))
            vars_map = view_gray.var(axis=(2, 3))
            
            v_TL, v_TR = vars_map[0:H, 0:W], vars_map[0:H, radius:W+radius]
            v_BL, v_BR = vars_map[radius:H+radius, 0:W], vars_map[radius:H+radius, radius:W+radius]
            
            all_vars = np.stack([v_TL, v_TR, v_BL, v_BR], axis=-1)
            min_var_idx = np.argmin(all_vars, axis=-1)
            
            channels = img_float.shape[-1] if img_float.ndim == 3 else 1
            result = np.zeros_like(img_float)
            
            for c in range(channels):
                img_c = pad_image[..., c] if img_float.ndim == 3 else pad_image
                view_c = sliding_window_view(img_c, (k, k))
                means_c = view_c.mean(axis=(2, 3))
                
                m_list = [means_c[0:H, 0:W], means_c[0:H, radius:W+radius], 
                          means_c[radius:H+radius, 0:W], means_c[radius:H+radius, radius:W+radius]]
                
                res_c = np.take_along_axis(np.stack(m_list, axis=-1), min_var_idx[..., np.newaxis], axis=-1)[..., 0]
                
                if img_float.ndim == 3:
                    result[..., c] = res_c
                else:
                    result = res_c

            processed_data = result

        # ---------------------------------------------------------
        # 6. SPECTRAL RESIDUAL SALIENCY (EYE TRACKING)
        # ---------------------------------------------------------
        elif operation == "6. AI Saliency (Eye Tracking)":
            if img_float.ndim == 3:
                gray = np.dot(img_float[..., :3], [0.299, 0.587, 0.114])
            else:
                gray = img_float

            fft_img = np.fft.fft2(gray)
            amplitude = np.abs(fft_img)
            phase = np.angle(fft_img)

            log_amplitude = np.log(amplitude + 1e-8)

            pad_log = np.pad(log_amplitude, 1, mode='reflect')
            smooth_log = sliding_window_view(pad_log, (3, 3)).mean(axis=(2, 3))

            spectral_residual = log_amplitude - smooth_log

            reconstructed_fft = np.exp(spectral_residual + 1j * phase)
            saliency_map = np.abs(np.fft.ifft2(reconstructed_fft))
            
            saliency_map = saliency_map ** 2

            pad_sal = np.pad(saliency_map, 2, mode='reflect')
            smoothed_saliency = sliding_window_view(pad_sal, (5, 5)).mean(axis=(2, 3))

            s_min, s_max = smoothed_saliency.min(), smoothed_saliency.max()
            if s_max > s_min:
                smoothed_saliency = (smoothed_saliency - s_min) / (s_max - s_min) * 255.0

            processed_data = np.stack((smoothed_saliency,)*3, axis=-1) if image_data.ndim == 3 else smoothed_saliency

        # --- FINAL CLEANUP & SAFETY CHECK ---
        processed_data = np.clip(processed_data, 0, 255).astype(image_data.dtype)
        
        # Make sure metadata is a dictionary (just in case it comes in as None)
        if metadata is None:
            metadata = {}
            
        p_min, p_max = processed_data.min(), processed_data.max()
        if p_min == p_max:
            metadata['contrast_limits'] = (float(p_min), float(p_max) + 0.0001)
        else:
            metadata['contrast_limits'] = (float(p_min), float(p_max))

        return processed_data