from cv2 import cv2
import numpy as np

def thinning(filepath):
    img = cv2.imread(filepath, 0)

    img = 255 - img
    img1 = img.copy()

    # # Create an empty output image to hold values
    # # thin = np.zeros(img.shape, dtype='uint8')
    # # thin = cv2.ximgproc.thinning(img, thin)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, dilate_ker, iterations=1)
    # # img = cv2.erode(img, dilate_ker, iterations=1)
    # img = cv2.blur(img, (5, 5))
    # filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # img = cv2.filter2D(img, -1, filter)
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # img = cv2.blur(img, (3, 3))
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
    # ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    # img1 = img.copy()
    # img = cv2.addWeighted(img, 2, img1, -0.8, 0, img1)
    img = 255 - img
    # img1 = cv2.dilate(img1, dilate_ker, iterations=1)
    # Loop until erosion leads to an empty set
    # while (cv2.countNonZero(img1) != 0):
    #     # Erosion
    #     erode = cv2.erode(img1, kernel)
    #     # Opening on eroded image
    #     opening = cv2.morphologyEx(erode, cv2.MORPH_OPEN, kernel)
    #     # Subtract these two
    #     subset = erode - opening
    #     # Union of all previous sets
    #     thin = cv2.bitwise_or(subset, thin)
    #     # Set the eroded image for next iteration
    #     img1 = erode.copy()
    return img
