import numpy as np
from PIL import Image

def convert_to_grayscale(image_path):
    image = Image.open(image_path).convert("RGB")
    arr = np.array(image).astype(np.float32)

    # luminance grayscale: 0.299R + 0.587G + 0.114B
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray

if __name__ == "__main__":
    gray = convert_to_grayscale("test.jpg")
    out = Image.fromarray(gray, mode="L")
    out.save("output_gray.jpg")
    print("Saved output_gray.jpg")