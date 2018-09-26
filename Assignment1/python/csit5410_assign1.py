#!/usr/bin/python

import cv2
import numpy as np
from skimage import io

def im2double(img):
    result = np.zeros(img.shape, np.float64)
    height, width = result.shape
    for i in range(height):
        for j in range(width):
            result[i, j] = img[i, j] / 255.

    return result

def get_max(img):
    result = 0.0
    for column in img:
        for intansity in column:
            if intansity > result:
                result = intansity

    return result

def get_min(img):
    result = 1.0
    for column in img:
        for intansity in column:
            if intansity < result:
                result = intansity

    return result

def get_mean(img, T):
    sum_larger = 0
    sum_smaller = 0
    count_larger = 0
    count_smaller = 0
    for column in img:
        for intansity in column:
            if intansity >= T:
                sum_larger += intansity
                count_larger += 1
            else:
                sum_smaller += intansity
                count_smaller += 1

    return 0.5 * ((sum_larger + 0.0) / count_larger + (sum_smaller + 0.0) / count_smaller)

def myprewittedge(Im, T, direction):
    height, width = Im.shape
    g = np.zeros(Im.shape, np.uint8)

    if T == None:
        T = 0.5 * (get_max(Im) + get_min(Im))
        previous = T
        for i in range(10):
            T = get_mean(Im, T)
            if 0.05 * previous > abs(T - previous):
                break
            previous = T

    filter1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    filter2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter3 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    filter4 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            curr_sub_region = np.array([[Im[i - 1, j - 1], Im[i - 1, j], Im[i - 1, j + 1]],
                                        [Im[i, j - 1],     Im[i, j],     Im[i, j + 1]],
                                        [Im[i + 1, j - 1], Im[i + 1, j], Im[i + 1, j + 1]]])
            temp = np.sum(filter1 * curr_sub_region)
            if abs(temp) >= T:
                g[i, j] = 255
                continue

            temp = np.sum(filter2 * curr_sub_region)
            if abs(temp) >= T:
                g[i, j] = 255
                continue
            
            temp = np.sum(filter3 * curr_sub_region)
            if abs(temp) >= T:
                g[i, j] = 255
                continue

            temp = np.sum(filter4 * curr_sub_region)
            if abs(temp) >= T:
                g[i, j] = 255
    
    return g

def Task1():
    Im = cv2.imread('./fig.tif', cv2.IMREAD_UNCHANGED)
    # cv2.imshow('./01original.jpg', Im)
    cv2.imwrite('./01original.jpg', Im)
    # cv2.waitKey(2000)
    return Im

def Task2(Im):
    threshold = get_max(Im) * 0.2
    g = myprewittedge(Im, threshold, "all")
    cv2.imshow('./02binary1.jpg', g)
    cv2.imwrite('./02binary1.jpg', g)
    cv2.waitKey(2000)
    return g

def Task3(Im):
    f = myprewittedge(Im, None, "all")
    cv2.imshow('./03binary2.jpg', f)
    cv2.imwrite('./03binary2.jpg', f)
    cv2.waitKey(2000)
    return f

def main():
    Im = Task1()
    Im = im2double(Im)
    g = Task2(Im)
    f = Task3(Im)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()