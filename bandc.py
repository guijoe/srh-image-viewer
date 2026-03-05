import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("straw.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert to float for safe processing
img_float = img.astype(np.float32)

# Contrast (alpha) and brightness (beta)
alpha = 1.5  # Contrast control (1.0-3.0)
beta = 30  # Brightness control (0-100)

enhanced = alpha * img_float + beta

# Clip values to valid range
enhanced = np.clip(enhanced, 0, 255)
enhanced = enhanced.astype(np.uint8)

# Show result
plt.imshow(enhanced)
plt.title("Brightness and Contrast")
plt.axis("off")
plt.show()