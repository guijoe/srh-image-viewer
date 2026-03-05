import numpy as np

def contrast_stretch(image):
    img = image.astype(np.float32)
    min_val = img.min()
    max_val = img.max()
    stretched = (img - min_val) / (max_val - min_val) * 255
    return stretched.astype(np.uint8)
  
