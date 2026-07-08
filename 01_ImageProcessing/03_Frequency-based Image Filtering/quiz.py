import cv2
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

image = cv2.imread('./02_cv3/data/Lena.png', 0).astype(np.float32) / 255

#Image Filtering
#(1)
KSIZE = 11
ALPHA = 2
kernel_unsharp = cv2.getGaussianKernel(KSIZE, 0)
kernel_unsharp = -ALPHA * kernel_unsharp @ kernel_unsharp.T
kernel_unsharp[KSIZE//2, KSIZE//2] += 1 + ALPHA
filtered = cv2.filter2D(image, -1, kernel_unsharp)
cv2.imshow('Unsharp Mask', filtered)
cv2.waitKey()
cv2.destroyAllWindows()

#(2)
dx = cv2.Sobel(filtered, cv2.CV_32F, 1, 0)
cv2.imshow('Sobel Filter', dx)
cv2.waitKey()
cv2.destroyAllWindows()

#(3)
kernel_gabor = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
kernel_gabor /= math.sqrt((kernel_gabor * kernel_gabor).sum())
filtered = cv2.filter2D(filtered, -1, kernel_gabor)
cv2.imshow('Gabor Filter', filtered)
cv2.waitKey()
cv2.destroyAllWindows()

#(4)
fill_val = 0
mask = np.zeros((image.shape[0], image.shape[1]*2), np.uint8)
def trackbar_callback(value):
    global fill_val
    fill_val = value
cv2.namedWindow('Thresholded')
cv2.createTrackbar('Threshold', 'Thresholded', fill_val, 255, lambda v: trackbar_callback(v))
while True:
    thr_1, mask_1 = cv2.threshold(dx, fill_val, 1, cv2.THRESH_BINARY)
    thr_2, mask_2 = cv2.threshold(filtered, fill_val, 1, cv2.THRESH_BINARY)
    mask = np.hstack((mask_1, mask_2))
    cv2.imshow('Thresholded', mask)
    key = cv2.waitKey(3)
    if key == 27:  # ESC key
        break
cv2.destroyAllWindows()

#(5)
kernel_morph = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_morph)
closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_morph)
result = np.vstack((opening, closing))
cv2.imshow('Morphological Operations', result)
cv2.waitKey()
cv2.destroyAllWindows()



#Frequency-based Filtering
#(1)
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
shifted = np.fft.fftshift(fft, axes=[0, 1])
magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
magnitude = np.log(magnitude)
restored = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
cv2.imshow('DFT Restored', restored)
cv2.waitKey()
cv2.destroyAllWindows()

#(2)Rectangle Mask
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
sz = 25
mask = np.zeros(fft.shape, np.uint8)
center = (image.shape[1]//2, image.shape[0]//2)
cv2.rectangle(mask, (center[0]-sz, center[1]-sz), (center[0]+sz, center[1]+sz), (1, 1), -1)
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
cv2.imshow('Rectangle Mask Filtered', filtered)
cv2.waitKey()
cv2.destroyAllWindows()

#(3)Circle Mask
fft = cv2.dft(image, flags=cv2.DFT_COMPLEX_OUTPUT)
fft_shift = np.fft.fftshift(fft, axes=[0, 1])
mask = np.zeros(fft.shape, np.uint8)
center = (image.shape[1]//2, image.shape[0]//2)
cv2.circle(mask, center, 25, (1, 1), -1)
fft_shift *= mask
fft = np.fft.ifftshift(fft_shift, axes=[0, 1])
filtered = cv2.idft(fft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
cv2.imshow('Circle Mask Filtered', filtered)
cv2.waitKey()
cv2.destroyAllWindows()

