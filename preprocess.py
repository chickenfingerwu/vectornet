from cv2 import cv2
import numpy as np


def thinning(img, img_domain, is_gt):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = img.shape
    scale_percent_h = (64 / height) * 100
    scale_percent_w = (64 / width) * 100
    if not is_gt:
        cv2.imwrite("current_fake_result_%s.png" % img_domain, img)
    average = np.average(img)
    if not img_domain == 'A' or not is_gt:
        if average >= 255 / 2:
            ret, img = cv2.threshold(img, average - (0.5 * (255 - average)), 255, cv2.THRESH_BINARY)
            img = 255 - img
        else:
            ret, img = cv2.threshold(img, average + (0.5 * (255 - average)), 255, cv2.THRESH_BINARY)
    if not is_gt:
        cv2.imwrite("current_fake_result_1_%s_threshold.png" % img_domain, img)
    # if is_gt:
    #     img = 255 - img
    if is_gt and img_domain == 'A':
        img = 255 - img
        kernel = np.ones(3, dtype='uint8')
        img = cv2.erode(img, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.resize(img, (64, 64), fx=scale_percent_w, fy=scale_percent_h)
    if not is_gt:
        cv2.imwrite("current_fake_result_2_%s_resize.png" % img_domain, img)

    # img = 255 - img
    # img = cv2.ximgproc.thinning(img)

    return img

img = cv2.imread('047584.png')
img = 255 - thinning(img, 'A', True)
cv2.imshow('sth', img)
cv2.waitKey(0)