import cv2
import numpy as np
import matplotlib.pyplot as plt


#Matrix Manipulating
image = np.full((480, 640, 3), 255, dtype=np.uint8)
cv2.imshow('White Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = np.zeros((480, 640, 3), (0,0,255), dtype=np.uint8)
cv2.imshow('Red Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image.fill(0)
cv2.imshow('Black Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Converting data type
image = cv2.imread('./data/Lena.png')
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

image = image.astype(np.float32) / 255.0
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', np.clip(image * 2, 0, 1))
cv2.waitKey(0)
cv2.destroyAllWindows()

image = (image * 255).astype(np.uint8)
print('Shape:', image.shape)
print('Data type:', image.dtype)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Manipulation image channels
image = cv2.imread('./data/Lena.png')
print('Shape:', image.shape)
cv2.imshow('Original Image', image)

image[:, :, [0,2]] = image[:,:,[2,0]]
cv2.imshow('blue_and_red_swapped', image)
cv2.waitKey(0)

image[:, :, [0,2]] = image[:,:,[2,0]]
image[:, :, 0] = (image[:, :, 0] * 0.9).clip(0, 1)
image[:,:,1] = (image[:,:,1] * 1.1).clip(0, 1)
cv2.imshow('converted_image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Converting color spaces
image = cv2.imread('./data/Lena.png').astype(np.float32) / 255.0
print('Shape:', image.shape)
print('Data type:', image.dtype)
cv2.imshow('Original Image', image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print('Converted to Grayscale')
print('Shape:', gray.shape)
print('Data type:', gray.dtype)
cv2.imshow('Grayscale Image', gray)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
print('Converted to HSV')
print('Shape:', hsv.shape)
print('Data type:', hsv.dtype)

hsv[:,:,2] *= 2
from_hsv = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
print('Converted back to BGR from HSV')
print('Shape:', from_hsv.shape)
print('Data type:', from_hsv.dtype)
cv2.imshow('From HSV Image', from_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Gamma correction
image = cv2.imread('./data/Lena.png').astype(np.float32) / 255.0
gamma = 0.5
corrected_image = np.power(image, gamma)

cv2.imshow('Original Image', image)
cv2.imshow('Gamma Corrected Image', corrected_image)
cv2.waitKey(0)

cv2.imwrite('./tmp/image.png', (image * 255))
cv2.imwrite('./tmp/corrected_image.png', (corrected_image * 255))
cv2.destroyAllWindows()

