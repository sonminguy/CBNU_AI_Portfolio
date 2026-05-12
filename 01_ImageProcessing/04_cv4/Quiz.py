import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('./data/Lena.png', 0)


# 1------------------------------------------------------------------------------------------
otsu_thr, otsu_mask = cv2.threshold(image, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
print('Otsu Threshold:', otsu_thr)

plt.figure(figsize=(6, 3))
plt.subplot(121)
plt.axis('off')
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.axis('off')
plt.title('Otsu Mask')
plt.imshow(otsu_mask, cmap='gray')
plt.tight_layout()
plt.show()


# 2------------------------------------------------------------------------------------------
_, contours, hierarchy = cv2.findContours(otsu_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

image_external = np.zeros(otsu_mask.shape, otsu_mask.dtype)
for i in range(len(contours)):
    if hierarchy[0][i][3] == -1:
        cv2.drawContours(image_external, contours, i, 255, -1)

image_internal = np.zeros(otsu_mask.shape, otsu_mask.dtype)
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


# 3_random connected components------------------------------------------------------------------------------------------
connectivity = 0
press_enter = False
press_exit = False
while not press_enter:
    num_labels, labelmap, stats, centers = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

    colored = np.full((image.shape[0], image.shape[1], 3), 0, dtype=np.uint8)
    for l in range(1, num_labels):
        if stats[l][4] > 200:
            colored[labelmap == l] = (0, 255 * l / num_labels, 255 * (num_labels - l) / num_labels)
            cv2.circle(colored, (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imshow('Connected Components with Otsu Thresholding', np.hstack((img, colored)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    key = cv2.waitKey(0)
    if key == 13:  # Enter key
        press_enter = True
    elif key == 27:  # Escape key
        press_exit = True
        break