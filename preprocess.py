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
        percent_h = (256 / height) * 100
        percent_w = (256 / width) * 100
        img = cv2.resize(img, (256, 256), fx=percent_w, fy=percent_h)
        # kernel = np.ones(5, dtype='uint8')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        img = cv2.erode(img, kernel, iterations=3)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.resize(img, (64, 64), fx=scale_percent_w, fy=scale_percent_h)
    if not is_gt:
        cv2.imwrite("current_fake_result_2_%s_resize.png" % img_domain, img)

    # img = 255 - img
    # img = cv2.ximgproc.thinning(img)

    return img
