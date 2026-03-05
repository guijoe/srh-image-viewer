import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("straw.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Sharpening kernel
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

# Apply convolution
sharpened = cv2.filter2D(img, -1, kernel)

plt.imshow(sharpened)
plt.title("Sharpened Image")
plt.axis("off")
plt.show()