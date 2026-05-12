import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/BnW.png', 0)

_, contours, hierarchy = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(image.shape, image.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] != -1:
        cv2.drawContours(image_internal, contours, i, 255, -1)

plt.figure(figsize=(10, 3))
plt.subplot(131)
plt.axis('off')
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(132)
plt.axis('off')
plt.title('External Contours')
plt.imshow(image_external, cmap='gray')
plt.subplot(133)
plt.axis('off')
plt.title('Internal Contours')
plt.imshow(image_internal, cmap='gray')
plt.tight_layout()
plt.show()