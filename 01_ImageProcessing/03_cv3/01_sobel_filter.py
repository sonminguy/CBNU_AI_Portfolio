import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./02_cv3/data/Lena.png', 0)
if image is None:
    raise FileNotFoundError("Image not found. Please check the path.")
dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)

plt.figure(figsize=(8, 3))
plt.subplot(131)
plt.axis('off')
plt.title('image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title(r'$\frac{df}{dx}$')
plt.imshow(dx, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title(r'$\frac{df}{dy}$')
plt.imshow(dy, cmap='gray')
plt.tight_layout()
plt.show()