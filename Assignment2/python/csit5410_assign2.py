import math

import cv2
import numpy as np
import matplotlib.pyplot as plt

def myhoughcircle(Imbinary, r, thresh = 4):
    if thresh < 4:
        print('threshold value must be bigger or equal to 4')
        return

    h, w = Imbinary.shape
    Accumulator = np.zeros((h, w))
    boundaries = set()
    for y0 in range(h):
        for x0 in range(w):
            if Imbinary[y0, x0] == 255:
                boundaries.clear()
                angle = 0.0
                while angle < 2 * np.pi:
                    y = int(round(y0 + r * np.sin(angle)))
                    x = int(round(x0 + r * np.cos(angle)))
                    if (0 <= y) and (y < h):
                        if (0 <= x) and (x < w):
                            boundaries.add((y, x))
                    angle += np.pi / 180

                for pos in boundaries:
                    Accumulator[pos[0], pos[1]] += 1

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
                print('(%d, %d) -- %d' % (i , j, Accumulator[i, j]))
            if Accumulator[i, j] > max_value:
                max_y = i
                max_x = j
                max_value = Accumulator[i, j]

    if len(x0detect) == 0:
        y0detect = [max_y]
        x0detect = [max_x]

    return y0detect, x0detect, Accumulator

def draw_circle(img, Xc, Yc, r):
    h, w = img.shape
    for i in range(h):
        for j in range(w):
            if abs((j - Xc) * (j - Xc) + (i - Yc) * (i - Yc) - r * r) < 100:
                img[i, j] = 255

    return img

def myfld(input_sample, class1_samples, class2_samples):
    n1, n = class1_samples.shape
    n2, n = class2_samples.shape

    mean_c1 = np.sum(class1_samples, axis = 0).reshape(2, 1) / n1
    mean_c2 = np.sum(class2_samples, axis = 0).reshape(2, 1) / n2

    S1 = 0
    for i in range(n1):
        qi = class1_samples[i, :].reshape(2, 1)
        S1 += (qi - mean_c1) * (qi - mean_c1).T

    S2 = 0
    for i in range(n2):
        qi = class2_samples[i, :].reshape(2, 1)
        S2 += (qi - mean_c2) * (qi - mean_c2).T

    s_w = S1 + S2
    s_b = (mean_c2 - mean_c1) * (mean_c2 - mean_c1).T

    w = np.dot(np.linalg.inv(s_w), (mean_c2 - mean_c1))
    seperation_point = np.dot(0.5 * w.T, (mean_c2 - mean_c1))
    value = np.dot(input_sample, w)

    output_class = 1
    if value >= seperation_point:
        output_class = 2

    return output_class, w, s_w, mean_c1, mean_c2

def main():
    print('*********************************************')
    print('Task 1: Hough Circle Detection')
    print('*********************************************')

    I = cv2.imread('./qiqiu.png')
    plt.figure('Figure 1')
    plt.imshow(I)
    plt.axis('off')
    plt.title('Figure 1')
    plt.show()
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = cv2.Canny(I, 30, 100)
    plt.figure('Figure 2')
    plt.imshow(I)
    plt.axis('off')
    plt.title('Figure 2')
    plt.show()
    radius = 114
    threshold = 110
    y0detect, x0detect, Accumulator = myhoughcircle(I, radius, threshold)
    plt.figure('Figure 3')
    plt.imshow(Accumulator)
    plt.axis('on')
    plt.title('Figure 3')
    plt.show()

    img = np.zeros(I.shape)
    for i in range(len(y0detect)):
        x = x0detect[i]
        y = y0detect[i]
        img = draw_circle(img, x, y, radius)

    plt.figure('Figure 4')
    plt.imshow(img)
    plt.axis('off')
    plt.title('Figure 4')
    plt.show()

    print('*********************************************')
    print('Task 2: Fisher Linear Discriminant')
    print('*********************************************')

    class1_samples = np.array([[1, 2], [2, 3], [3, 3], [4, 5], [5, 5]])
    class2_samples = np.array([[1, 0], [2, 1], [3, 1], [3, 2], [5, 3], [6, 5]])
    input_sample = np.array([2, 5])
    output_class, w, s_w, mean_c1, mean_c2 = myfld(input_sample, class1_samples, class2_samples)
    print('Output class is %d.' % output_class)
    print('Within-class scatter matrix:')
    print(s_w)
    print('Weights:')
    print(w)

if __name__ == '__main__':
    main()