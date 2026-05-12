import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
parser.add_argument('--out_png', default='./data/lena_draw.png', help='Output PNG file name.')
parser.add_argument('--iter', default=50, type=int, help='Downsampling-Upsampling iteration number')
parms = parser.parse_args()
orig = cv2.imread(parms.path)

#Lena 컬러 이미지 출력
lena = orig.copy()
cv2.imshow('Lena', lena)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Lena 그레이스케일 이미지 출력
lena_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
cv2.imshow('Lena Gray', lena_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Lena Histogram Equalization
lena_hist_eq = cv2.equalizeHist(lena_gray)
cv2.imshow('Lena Histogram Equalization', lena_hist_eq)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Lena Gamma Correction
gamma = 2.2
lena_gamma = np.power(lena_gray / 255.0, gamma) * 255.0
lena_gamma = lena_gamma.astype(np.uint8)
cv2.imshow('Lena Gamma Correction', lena_gamma)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Lena HSV H Median Filter, S Gaussian Filter, V Bilateral Filter
lena_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(lena_hsv)
h_median = cv2.medianBlur(h, 5)
s_gaussian = cv2.GaussianBlur(s, (5, 5), 0)
v_bilateral = cv2.bilateralFilter(v, 9, 75, 75)
lena_h_midian = cv2.merge((h_median, s, v))
lena_s_gaussian = cv2.merge((h, s_gaussian, v))
lena_v_bilateral = cv2.merge((h, s, v_bilateral))
lena_h_median_bgr = cv2.cvtColor(lena_h_midian, cv2.COLOR_HSV2BGR)
lena_s_gaussian_bgr = cv2.cvtColor(lena_s_gaussian, cv2.COLOR_HSV2BGR)
lena_v_bilateral_bgr = cv2.cvtColor(lena_v_bilateral, cv2.COLOR_HSV2BGR)
cv2.imshow('Lena H Median Filter', lena_h_median_bgr)
cv2.imshow('Lena S Gaussian Filter', lena_s_gaussian_bgr)
cv2.imshow('Lena V Bilateral Filter', lena_v_bilateral_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()