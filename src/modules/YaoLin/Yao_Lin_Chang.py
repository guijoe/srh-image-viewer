import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def invert_colors(image):
    """
    Invert all colors in the image.
    Each pixel value becomes 255 - value.
    """

    img = image.copy()

    # invert colors
    img = 255 - img

    return img


# load image
img = Image.open("test.jpg")
img_array = np.array(img)

# apply inversion
result = invert_colors(img_array)

# show before and after
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img_array)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Inverted Colors")
plt.imshow(result)
plt.axis("off")

plt.show()

# save output
Image.fromarray(result).save("inverted_image.jpg")

print("Saved output as inverted_image.jpg")