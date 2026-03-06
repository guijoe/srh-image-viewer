import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def extract_blue(image):
    """
    Keep blue areas colored.
    Everything else becomes grayscale.
    """

    img = image.copy()

    # convert image to grayscale
    gray = np.mean(img, axis=2).astype(np.uint8)
    gray_rgb = np.stack([gray, gray, gray], axis=2)

    # detect blue dominant pixels
    blue_mask = (img[:, :, 2] > img[:, :, 0]) & (img[:, :, 2] > img[:, :, 1])

    # start with grayscale
    result = gray_rgb.copy()

    # keep original color where blue dominates
    result[blue_mask] = img[blue_mask]

    return result


# load image
img = Image.open("test.jpg")
img_array = np.array(img)

# apply filter
result = extract_blue(img_array)

# show images
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img_array)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("Blue Highlight")
plt.imshow(result)
plt.axis("off")

plt.tight_layout()
plt.show()

# save output
Image.fromarray(result).save("blue_highlight.jpg")

print("Saved output as blue_highlight.jpg")