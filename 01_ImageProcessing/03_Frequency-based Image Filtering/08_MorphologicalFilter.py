import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./02_cv3/data/Lena.png', 0)

_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

eroded = cv2.erode(binary, cv2.MORPH_ERODE, (3, 3), iterations=10)
dilated = cv2.dilate(binary, cv2.MORPH_DILATE, (3, 3), iterations=10)

opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

plt.figure(figsize=(10, 10))
plt.subplot(221)
plt.axis('off')
plt.title('Binary')
plt.imshow(binary, cmap='gray')
plt.subplot(222)
plt.axis('off')
plt.title('Eroded')
plt.imshow(eroded, cmap='gray')
plt.subplot(223)
plt.axis('off')
plt.title('Dilated')
plt.imshow(dilated, cmap='gray')
plt.subplot(224)
plt.axis('off')
plt.title('Opened')
plt.imshow(opened, cmap='gray')
plt.subplot(224)
plt.axis('off')
plt.title('Closed')
plt.imshow(closed, cmap='gray')
plt.subplot(224)
plt.axis('off')
plt.title('Gradient')
plt.imshow(gradient, cmap='gray')

plt.tight_layout(True)
plt.show()