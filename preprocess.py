from cv2 import cv2
import numpy as np

def thinning(img):
    height, width = img.shape
    scale_percent_h = (64 / height) * 100
    scale_percent_w = (64 / width) * 100

    img = 255 - img
    ret, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.resize(img, (64, 64), fx=scale_percent_w, fy=scale_percent_h)

    img = cv2.ximgproc.thinning(img)
    # img = 255 - img

    return img


# img = thinning("047619_fake.png")
# cv2.imshow("sth", img)
# cv2.imwrite("047584_process.png", img)
# cv2.waitKey(0)
