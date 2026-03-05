import numpy as np

class ImageDataStore:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ImageDataStore, cls).__new__(cls)
            cls._instance.current_image_data = None
            cls._instance.current_image_metadata = {}
            cls._instance.image_session_id = None # Unique ID for the current image
        return cls._instance

    def set_image(self, data: np.ndarray, metadata: dict = None, session_id: str = None):
        self.current_image_data = data
        self.current_image_metadata = metadata if metadata is not None else {}
        self.image_session_id = session_id if session_id is not None else str(id(data))
        print(f"Image data updated. Shape: {data.shape}, Dtype: {data.dtype}, Session ID: {self.image_session_id}")

    def get_image(self):
        return self.current_image_data, self.current_image_metadata, self.image_session_id

    def clear_image(self):
        self.current_image_data = None
        self.current_image_metadata = {}
        self.image_session_id = None