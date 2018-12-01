import math
import numpy as np

def display_2(matrix):
    if len(matrix.shape) < 2:
        n = matrix.shape[0]
        for i in range(n):
            if matrix[i] < 0.0:
                print('%.2f' % matrix[i], end = ' ')
            else:
                print(' %.2f' % matrix[i], end = ' ')
            print()
        print()
    else:
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                if matrix[i, j] < 0.0:
                    print('%.2f' % matrix[i, j], end = ' ')
                else:
                    print(' %.2f' % matrix[i, j], end = ' ')
            print()
        print()

def display_4(matrix):
    if len(matrix.shape) < 2:
        n = matrix.shape[0]
        for i in range(n):
            if matrix[i] < 0.0:
                print('%.4f' % matrix[i], end = ' ')
            else:
                print(' %.4f' % matrix[i], end = ' ')
            print()
        print()
    else:
        n, m = matrix.shape
        for i in range(n):
            for j in range(m):
                if matrix[i, j] < 0.0:
                    print('%.4f' % matrix[i, j], end = ' ')
                else:
                    print(' %.4f' % matrix[i, j], end = ' ')
            print()
        print()

def main():
    '''
        Part 1
    '''
    x1 = np.array([2, 1, 5])
    x2 = np.array([3, 2, 7])
    x3 = np.array([2, 6, 6])

    mean_x = (x1 + x2 + x3) / 3

    print('Mean image vector = ')
    display_4(mean_x)

    temp2 = np.array([x1 - mean_x, x2 - mean_x, x3 - mean_x])
    temp1 = temp2.T
    display_4(temp1)
    display_4(temp2)

    S = np.dot(temp1, temp2)
    print('Scatter matrix = ')
    display_4(S)

    '''
        Part 2
    '''
    sigma1 = 0
    e1 = np.array([-0.8701, -0.0967, 0.4834])
    sigma2 = 2.5215
    e2 = np.array([-0.4882, 0.0334, -0.8721])
    sigma3 = 14.1452
    e3 = np.array([-0.0681, 0.9948, 0.0763])
    x4 = np.array([2, 2, 4])

    eigens = [e3, e2] # using non-zero eigenvalues
    faces = [x1, x2, x3]

    approximated_x4 = mean_x.copy()
    for e in eigens:
        approximated_x4 += np.dot((x4 - mean_x), e) * e

    print('Approximation for face 4 = ')
    display_2(approximated_x4)

    for i in range(len(faces)):
        face = faces[i]
        distance = 0.0
        delta = face - approximated_x4
        for j in range(delta.shape[0]):
            distance += delta[j] ** 2
        distance = math.sqrt(distance)
        print('Distance between x%d and x4 = ' % (i + 1))
        print('%.2f' % distance)

if __name__ == '__main__':
    main()