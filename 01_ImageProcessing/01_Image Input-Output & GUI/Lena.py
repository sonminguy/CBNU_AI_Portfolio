import argparse
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--path', default='./data/Lena.png', help='Image path.')
parser.add_argument('--out_png', default='./data/lena_draw.png', help='Output PNG file name.')
parser.add_argument('--iter', default=50, type=int, help='Downsampling-Upsampling iteration number')
parms = parser.parse_args()
orig = cv2.imread(parms.path)
img_show = orig.copy()
orig_size = orig.shape[:2]

finish = False
mouse_pressed = False
s_x = s_y = e_x = e_y = -1


def rand_pt():
    return (random.randrange(orig_size[1]), random.randrange(orig_size[0]))

def mouse_callback(event, x, y, flags, param):
    global img_show, s_x, s_y, e_x, e_y, mouse_pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_pressed = True
        s_x, s_y = x, y
        image_show = np.copy(orig)
    elif event == cv2.EVENT_MOUSEMOVE:
        if mouse_pressed:
            image_show = np.copy(orig)
            cv2.rectangle(image_show, (s_x, s_y), (x, y), (0, 255, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        mouse_pressed = False
        e_x, e_y = x, y


cv2.namedWindow('result')
cv2.setMouseCallback('result', mouse_callback)

while not finish:
    cv2.imshow('result', img_show)
    key = cv2.waitKey(1)
    if key == ord('r'):
        cv2.rectangle(img_show, rand_pt(),rand_pt(),(255,0,0),-1)
    elif key == ord('l'):
        cv2.line(img_show, rand_pt(), rand_pt(), (0, 255, 0), 3)
    elif key == ord('a'):
        cv2.arrowedLine(img_show, rand_pt(), rand_pt(), (0, 0, 255), 3)
    elif key == ord('w'):
        cv2.imwrite(parms.out_png, img_show, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    elif key == ord('c'):
        
        img_show = orig.copy()
    elif key == 27:  # ESC key
        finish = True
cv2.destroyAllWindows()