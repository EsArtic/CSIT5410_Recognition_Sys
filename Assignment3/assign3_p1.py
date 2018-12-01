import math
import numpy as np

N = 6

def display(dictionary):
    for i in range(N):
        print('%.2f' % dictionary[i], end = ' ')
    print('\n')

def main():
    img = [[1, 5, 4, 1, 1, 3],
           [0, 1, 5, 5, 4, 0],
           [5, 4, 4, 0, 2, 3],
           [2, 5, 0, 0, 1, 1],
           [5, 2, 3, 5, 2, 4],
           [5, 0, 4, 3, 5, 2]]

    hist_freq = {}
    for i in range(N):
        for j in range(N):
            if img[i][j] in hist_freq.keys():
                hist_freq[img[i][j]] += 1
            else:
                hist_freq[img[i][j]] = 1

    print('hist_freq = ')
    display(hist_freq)

    p_i = {}
    for i in range(N):
        p_i[i] = hist_freq[i] / (N * N)
    print('p_i = ')
    display(p_i)

    P1 = {}
    for i in range(N):
        curr = 0.0
        for j in range(i + 1):
            curr += p_i[j]
        P1[i] = curr
    print('P1 = ')
    display(P1)

    m = {}
    for i in range(N):
        curr = 0.0
        for j in range(i + 1):
            curr += j * p_i[j]
        m[i] = curr
    print('m = ')
    display(m)

    m_G = m[N - 1]
    print('m_G = %.2f\n' % m_G)

    sigma_b = {}
    for i in range(N):
        sigma_b[i] = ((m_G * P1[i] - m[i]) ** 2) / (P1[i] * (1 - P1[i]))
    print('sigma_b = ')
    display(sigma_b)

    temp = sigma_b.items()
    threshold = sorted(temp, key = lambda x: x[1])[-1][1]

    output = np.zeros((N, N), dtype = int)
    for i in range(N):
        for j in range(N):
            if img[i][j] >= threshold:
                output[i, j] = 1
            else:
                output[i, j] = 0

    print('Thresholded image = ')
    print(output)

if __name__ == '__main__':
    main()