import cv2
import numpy as np

image = cv2.imread('./data/BnW.png', 0)

connectivity = 0
num_labels, labelmap, stats, centers = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

img = np.hstack((image, labelmap.astype(np.float32) / (num_labels - 1)))
cv2.imshow('Connected Components', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('./data/Lena.png', 0)
otsu_threshold, otsu_mask = cv2.threshold(img, -1, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

output = cv2.connectedComponentsWithStats(otsu_mask, connectivity, cv2.CV_32S)

num_labels, labelmap, stats, centers = output

colored = np.full((img.shape[0], img.shape[1], 3), 0, dtype=np.uint8)

for l in range(1, num_labels):
    if stats[l][4] > 200:
        colored[labelmap == l] = (0, 255 * l / num_labels, 255 * (num_labels - l) / num_labels)
        cv2.circle(colored, (int(centers[l][0]), int(centers[l][1])), 5, (255, 0, 0), cv2.FILLED)

img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

cv2.imshow('Connected Components with Otsu Thresholding', np.hstack((img, colored)))
cv2.waitKey(0)
cv2.destroyAllWindows()