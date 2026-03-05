import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("straw.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Normalize image to 0–1
img_norm = img / 255.0

gamma = 2 # <1 brighter, >1 darker

gamma_corrected = np.power(img_norm, gamma)

gamma_corrected = (gamma_corrected * 255).astype(np.uint8)

plt.imshow(gamma_corrected)
plt.title("Gamma Corrected Image")
plt.axis("off")
plt.show()