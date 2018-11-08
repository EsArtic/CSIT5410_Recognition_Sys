import math

import cv2
import numpy as np

def myhoughcircle(Imbinary, r, thresh = 4):
    if thresh < 4:
        print('threshold value must be bigger or equal to 4')
        return

    h, w = Imbinary.shape
    Accumulator = np.zeros((h, w))
    for i0 in range(h):
        for j0 in range(w):
            if Imbinary[i0, j0] == 255:
                for i1 in range(max(0, i0 - r), min(h, i0 + r + 1)):
                    delta = math.sqrt(r * r - (i1 - i0) * (i1 - i0))
                    if delta - int(delta) >= 0.5:
                        delta = int(delta) + 1
                    else:
                        delta = int(delta)

                    for n in range(-1, 2):
                        j11 = j0 + delta + n
                        j12 = j0 - delta + n
                        if j11 >= 0 and j11 < w:
                            Accumulator[i1, j11] += 1
                        if j12 >= 0 and j12 < w:
                            Accumulator[i1, j12] += 1

                    '''
                    j11 = j0 + delta
                    if j11 < w:
                        Accumulator[i1, j11] += 1
                    j12 = j0 - delta
                    if j12 >= 0:
                        Accumulator[i1, j12] += 1
                    '''

    max_value = Accumulator[0, 0]
    max_x = 0
    max_y = 0
    x0detect = []
    y0detect = []
    for i in range(h):
        for j in range(w):
            if Accumulator[i, j] >= thresh:
                y0detect.append(i)
                x0detect.append(j)
            if Accumulator[i, j] > max_value:
                max_y = i
                max_x = j
                max_value = Accumulator[i, j]

    if len(x0detect) == 0:
        y0detect = [max_y]
        x0detect = [max_x]

    # return y0detect, x0detect, Accumulator
    return [max_y], [max_x], Accumulator

def draw_circle(img, Xc, Yc, r):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if abs((i - Xc) * (i - Xc) + (j - Yc) * (j - Yc) - r * r) < 100:
                img[i, j] = 255

    return img

def main():
    print('*********************************************')
    print('Task 1: Hough Circle Detection')
    print('*********************************************')

    I = cv2.imread('../qiqiu.png')
    cv2.imshow('Figure 1', I)
    cv2.waitKey(1500)
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = cv2.Canny(I, 30, 150)
    cv2.imshow('Figure 2', I)
    cv2.waitKey(1500)
    radius = 114
    threshold = 110
    y0detect, x0detect, Accumulator = myhoughcircle(I, radius, threshold)
    cv2.imshow('Figure 3', Accumulator)
    cv2.waitKey(3000)

    img = np.zeros(I.shape)
    for i in range(len(y0detect)):
        x = x0detect[i]
        y = y0detect[i]
        img = draw_circle(img, x, y, radius)

    cv2.imshow('Figure 4', img)
    cv2.waitKey(1500)    

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()