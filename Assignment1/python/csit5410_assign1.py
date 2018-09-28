#!/usr/bin/python

import cv2
import numpy as np
import skimage.transform as st

def im2double(img):
    result = np.zeros(img.shape, np.float64)
    height, width = result.shape
    for i in range(height):
        for j in range(width):
            result[i, j] = img[i, j] / 255.

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

    # Algorithm 1: Automatically determined threshold
    if T == None:
        T = 0.5 * (np.max(Im) + np.min(Im))
        previous = T
        for i in range(10):
            T = get_mean(Im, T)
            if 0.05 * previous > abs(T - previous):
                break
            previous = T

    filter0 = None
    filter1 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    filter2 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    filter3 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]])
    filter4 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])

    if direction == 'all':
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
    else:
        if direction == 'horizontal':
            filter0 = filter1
        elif direction == 'vertical':
            filter0 = filter2
        elif direction == 'pos45':
            filter0 = filter3
        elif direction == 'neg45':
            filter0 = filter4

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                curr_sub_region = np.array([[Im[i - 1, j - 1], Im[i - 1, j], Im[i - 1, j + 1]],
                                            [Im[i, j - 1],     Im[i, j],     Im[i, j + 1]],
                                            [Im[i + 1, j - 1], Im[i + 1, j], Im[i + 1, j + 1]]])
                temp = np.sum(filter0 * curr_sub_region)
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
    threshold = np.max(Im) * 0.2
    g = myprewittedge(Im, threshold, "all")
    # cv2.imshow('./02binary1.jpg', g)
    cv2.imwrite('./02binary1.jpg', g)
    # cv2.waitKey(2000)
    return g

def Task3(Im):
    f = myprewittedge(Im, None, "all")
    # cv2.imshow('./03binary2.jpg', f)
    cv2.imwrite('./03binary2.jpg', f)
    # cv2.waitKey(2000)
    return f

def Task4(BW):
    # minLineLength = 200
    # maxLineGap = 15
    # lines = cv2.HoughLinesP(BW, 1, np.pi / 180, 118, minLineLength, maxLineGap)
    lines = st.probabilistic_hough_line(BW)

    Im = cv2.imread('./fig.tif')
    for line in lines:
        # for x1, y1, x2, y2 in line:
        #     cv2.line(Im, (x1, y1), (x2, y2), (255, 0, 0), 5)
        p0, p1 = line
        cv2.line(Im, (p0[0], p1[0]), (p0[1], p1[1]), (255, 0, 0), 5)

    cv2.imshow('houghline', Im)
    cv2.waitKey(10000)

def main():
    Im = Task1()
    print('Original image is read and displayed successfully.')
    Im = im2double(Im)
    g = Task2(Im)
    print('The corresponding binary edge image is computed and dispalyed successfully.')
    f = Task3(Im)
    print('The corresponding binary edge image is computed and displayed successfully.')
    Task4(f)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()