import argparse
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
parms = parser.parse_args()

cv2.namedWindow('window')

fill_val = np.array([255, 255, 255], np.uint8)

def trackbar_callback(idx, value):
    fill_val[idx] = value

cv2.createTrackbar('R', 'window', fill_val[0], 255, lambda v: trackbar_callback(2, v))
cv2.createTrackbar('G', 'window', fill_val[1], 255, lambda v: trackbar_callback(1, v))
cv2.createTrackbar('B', 'window', fill_val[2], 255, lambda v: trackbar_callback(0, v))

while True:
    image = np.full((500, 500, 3), fill_val)
    cv2.imshow('window', image)
    key = cv2.waitKey(3)
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()

