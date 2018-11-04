#!/usr/bin/python

import os
import sys
import math
import platform

import cv2
import numpy as np
import skimage.transform as st

ROOT = '../matlab/'
OUT_PATH = './'

def display(title, img):
    cv2.imshow(title, img)
    cv2.waitKey(1500)

def im2double(img):
    ret = np.zeros(img.shape, np.float64)
    height, width = ret.shape
    for i in range(height):
        for j in range(width):
            ret[i, j] = img[i, j] / 255.

    return ret

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

    return 0.5 * (float(sum_larger) / count_larger + float(sum_smaller) / count_smaller)

def myprewittedge(Im, T = None, direction = 'all'):
    '''
        Compute the binary edge image for the image Im.
        Args:
            Im: An intensity gray scale image.
            T: Threshold for generating the binary output image
            if T is not specified, automatic computed threshold will be provided.
            direction: A string for specifying whether to look for
            'horizontal' edges, 'vertical' edges, positive 45 degree 'pos45'
            edgs, negative 45 degree 'neg45' edges or 'all' edges.
        Returns:
            g: A binary image of the same size as Im, with 1's (255) where the
            function finds edges in Im and 0's eleswhere.
    '''
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

def get_top_n_peaks(peaks, num):
    result = []
    for i in range(min(num, len(peaks[0]))):
        result.append((peaks[0][i], peaks[1][i], peaks[2][i]))

    return result

def get_correspond_int(f):
    if f - int(f) >= 0.5:
        return int(f) + 1
    else:
        return int(f)

def search_for_lines(BW, peaks):
    '''
        Find lines in the binary edge image with the peaks computed by
        Hough line transform.
        Args:
            BW: A binary edge image.
            peaks: Peaks founded by Hough line transform in (hit_numbers,
            theta, distance) form.
        Returns:
            lines: A list containing the start point and end point for the
            founded lines.
    '''
    [row, col] = BW.shape
    lines = []

    for hits, angle, dist in peaks:
        dist = abs(dist)
        x0 = dist / np.cos(angle)
        b = dist / np.sin(angle)
        k = -1 * (b / x0)
        max_x = min(col - 1, get_correspond_int(x0))
        max_y = min(row - 1, get_correspond_int(b))
        min_x = max(0, get_correspond_int((max_y - b) / k))
        min_y = max(0, get_correspond_int(k * max_x + b))

        print('Current range:')
        print('[min_x, max_x] = [%d, %d]' % (min_x, max_x))
        print('[min_y, max_y] = [%d, %d]' % (min_y, max_y))
        print()

        if (max_y - min_y) > (max_x - min_x):
            start_point = (min_y, max_x)
            end_point = (min_y, max_x)
            start = False
            skip = 0
            for i in range(min_y, max_y + 1):
                xi = get_correspond_int((i - b) / k)
                edge = False
                for j in range(-2, 3):
                    if BW[i, xi + j] == 255:
                        edge = True
                        break
                if edge:
                    skip = 0
                    if not start:
                        start = True
                        start_point = (i, xi)
                else:
                    skip += 1
                    if start:
                        if skip > 10:
                            start = False
                            end_point = (i - 1, get_correspond_int((i - 1 - b) / k))
                            lines.append((start_point, end_point))
        else:
            start_point = (min_y, max_x)
            end_point = (min_y, max_x)
            start = False
            skip = 0
            for i in range(min_x, max_x + 1):
                yi = get_correspond_int(k * i + b)
                edge = False
                for j in range(-2, 3):
                    if BW[yi + j, i] == 255:
                        edge = True
                        break
                if edge:
                    skip = 0
                    if not start:
                        start_point = (yi, i)
                else:
                    skip += 1
                    if start:
                        if skip > 10:
                            start = False
                            end_point = get_correspond_int((k * (i - 1) + b), i - 1)
                            lines.append((start_point, end_point))

    return lines

def search_for_lines_interpolation(BW, peaks):
    '''
        Find lines in the binary edge image with the peaks computed by
        Hough line transform. Using bilinear interpolation to evaluate
        intansity on non-pixel postions.
        Args:
            BW: A binary edge image.
            peaks: Peaks founded by Hough line transform in (hit_numbers,
            theta, distance) form.
        Returns:
            lines: A list containing the start point and end point for the
            founded lines.
    '''
    [row, col] = BW.shape
    lines = []

    for hits, angle, dist in peaks:
        dist = abs(dist)
        x0 = dist / np.cos(angle)
        b = dist / np.sin(angle)
        k = -1 * (b / x0)
        max_x = min(col - 1, get_correspond_int(x0))
        max_y = min(row - 1, get_correspond_int(b))
        min_x = max(0, get_correspond_int((max_y - b) / k))
        min_y = max(0, get_correspond_int(k * max_x + b))

        print('Current range:')
        print('[min_x, max_x] = [%d, %d]' % (min_x, max_x))
        print('[min_y, max_y] = [%d, %d]' % (min_y, max_y))
        print()

        if (max_y - min_y) > (max_x - min_x):
            start_point = (min_y, max_x)
            end_point = (min_y, max_x)
            start = False
            skip = 0
            for i in range(min_y, max_y + 1):
                xi = (i - b) / k
                q1 = (i, int(xi) - 1)
                q2 = (i, int(xi) + 2)
                intansity = (q2[1] - xi) * BW[q1[0], q1[1]] + (xi - q1[1]) * BW[q2[0], q2[1]]
                edge = False
                if intansity >= 128.:
                    edge = True
                if edge:
                    skip = 0
                    if not start:
                        start = True
                        start_point = (i, get_correspond_int(xi))
                else:
                    skip += 1
                    if start:
                        if skip > 30:
                            start = False
                            end_point = (i - 1, get_correspond_int((i - 1 - b) / k))
                            lines.append((start_point, end_point))
        else:
            start_point = (min_y, max_x)
            end_point = (min_y, max_x)
            start = False
            skip = 0
            for i in range(min_x, max_x + 1):
                yi = k * i + b
                q1 = (int(yi) - 1, i)
                q2 = (int(yi) + 2, i)
                intansity = (q2[0] - yi) * BW[q1[0], q1[1]] + (yi - q1[0]) * BW[q2[0], q2[1]]
                edge = False
                if intansity >= 128.:
                    edge = True
                if edge:
                    skip = 0
                    if not start:
                        start_point = (get_correspond_int(yi), i)
                else:
                    skip += 1
                    if start:
                        if skip > 30:
                            start = False
                            end_point = get_correspond_int((k * (i - 1) + b), i - 1)
                            lines.append((start_point, end_point))

    return lines

def mylineextraction(BW):
    '''
        Extracts the longest line segment from the given binary image.
        Args:
            BW: A binary edge image.
        Returns:
            [sp, ep] = start point and end point of the longest line founded.
    '''
    H, theta, d = st.hough_line(BW)
    peaks = st.hough_line_peaks(H, theta, d)
    filtered_peaks = get_top_n_peaks(peaks, 5)

    print('Top %d peaks are:' % (len(filtered_peaks)))
    for item in filtered_peaks:
        print('Hits: %d, theta: %f, distance: %f' % (item[0], item[1], item[2]))

    # lines = search_for_lines(BW, filtered_peaks)
    lines = search_for_lines_interpolation(BW, filtered_peaks)

    max_sp = ()
    max_ep = ()
    max_length = 0
    print('Lines founded:')
    for i, line in enumerate(lines):
        sp, ep = line
        length = ((sp[0] - ep[0]) ** 2) * ((sp[1] - ep[1]) ** 2)

        print('start_point: %s, end_point: %s, length: %f' % (sp, ep, math.sqrt(length)))

        if length > max_length:
            max_sp = sp
            max_ep = ep
            max_length = length

    print('\nThe longest one:')
    print('start_point: %s, end_point: %s' % (max_sp, max_ep))

    return max_sp, max_ep

def sift(image):
    '''
        Python version for the given sift.m function
    '''
    cv2.imwrite('tmp.pgm', image)

    command = ''
    sysstr = platform.system()
    if sysstr == 'Windows':
        command = 'siftWin32'
    else:
        command = './sift'
    command += ' < tmp.pgm > tmp.key'
    os.system(command)

    g = open('tmp.key', 'r')
    header = g.readline().strip().split()
    if len(header) != 2:
        print('Error: Invalid keypoint file beginning.')
        return

    num = int(header[0])
    length = int(header[1])
    if length != 128:
        print('Error: Keypoint descriptor length invalid (should be 128).')

    locs = np.zeros((num, 4))
    descriptors = np.zeros((num, 128))
    for i in range(num):
        loc = g.readline().strip().split()
        for j in range(4):
            locs[i, j] = float(loc[j])

        des = []
        for j in range(7):
            des += g.readline().strip().split()
        for j in range(128):
            descriptors[i, j] = float(des[j])

    g.close()
    return descriptors, locs

def normalize(des):
    '''
        Noramlize the given SIFT descriptor by the formula:
        des[i] = des[i] / sqrt(des[0]^2 + des[1]^2 + ... + des[n]^2)
        Args:
            des: The initial SIFT descriptor
        Returns:
            ret: The normalized SIFT descriptor
    '''
    ret = np.zeros(des.shape)
    num, length = des.shape
    for i in range(num):
        len_sq = 0.0
        for j in range(length):
            len_sq += des[i, j] ** 2

        len_inv = 1.0 / math.sqrt(len_sq)
        for j in range(length):
            ret[i, j] = des[i, j] * len_inv

    return ret

def match(des1, des2):
    '''
       Python version for the given match.m function
    '''
    des1 = normalize(des1)
    des2 = normalize(des2)
    matches = []
    distRatio = 0.6
    des2t = des2.T
    num, length = des1.shape
    for i in range(num):
        dotprods = np.dot(des1[i, :].reshape(1, 128), des2t)
        dotprods = np.arccos(dotprods)
        mapper = {}
        for j in range(dotprods.shape[1]):
            mapper[j] = dotprods[0, j]
        mapper = sorted(mapper.items(), key = lambda x: x[1])
        if (mapper[0][1] < distRatio * mapper[1][1]):
            matches.append((i, mapper[0][0]))

    return matches

def screenmatches(I1, I2, matches, loc1match, des1match, loc2match, des2match):
    '''
        Python version for the given screenmatches.m function
    '''
    initial_len = len(matches[0])
    allScales = np.zeros((1, initial_len))
    allAngs = np.zeros((1, initial_len))
    allX = np.zeros((1, initial_len))
    allY = np.zeros((1, initial_len))
    for i in range(initial_len):
        print('Match %d: image 1 (scale, orient = %f, %f) matches, image2 (scale, orient = %f, %f)'
              % (i + 1, loc1match[i, 2], loc1match[i, 3], loc2match[i, 2], loc2match[i, 3]))
        scaleRatio = loc1match[i, 2] / loc2match[i, 2]
        dTheta = loc1match[i, 3] - loc2match[i, 3]

        # Force dTheta to be between -pi and +pi
        while dTheta > np.pi:
            dTheta -= 2 * np.pi
        while dTheta < -np.pi:
            dTheta += 2 * np.pi

        allScales[0, i] = scaleRatio
        allAngs[0, i] = dTheta

        # the feature in image 1
        x1 = loc1match[i, 0]
        y1 = loc1match[i, 1]

        # the feature in image 2
        x2 = loc2match[i, 0]
        y2 = loc2match[i, 1]

        '''
            The "center" of the object in image 1 is located at an offset of
            (-x1, -y1) relative to the detected feature. We need to scale and rotate
            this offset and apply it to the image2 location
        '''
        offset = np.array([-x1, -y1]).T
        offset = offset / scaleRatio
        # coefficients = np.array([[np.cos(dTheta), abs(np.sin(dTheta))], [-abs(np.sin(dTheta)), np.cos(dTheta)]])
        coefficients = np.array([[np.cos(dTheta), np.sin(dTheta)], [-np.sin(dTheta), np.cos(dTheta)]])
        offset = np.dot(coefficients, offset)

        allX[0, i] = x2 + offset[0]
        allY[0, i] = y2 + offset[1]

    '''
        Use a corase Hough space.
        Dimensions are [angle, scale, x, y]
        Define bin centers
    '''
    aBin = []
    i = -np.pi
    while i <= np.pi:
        aBin.append(i)
        i += np.pi / 4
    aBin = np.array(aBin)

    sBin = []
    i = 0.5
    while i <= 10:
        sBin.append(i)
        i += 2
    sBin = np.array(sBin)

    row, col = I2.shape
    xBin = []
    i = 0
    while i <= col - 1:
        xBin.append(i)
        i += col / 5
    xBin = np.array(xBin)

    yBin = []
    i = 0
    while i <= row - 1:
        yBin.append(i)
        i += row / 5
    yBin = np.array(yBin)

    H = np.zeros((len(aBin), len(sBin), len(xBin), len(yBin)))

    for i in range(initial_len):
        a = allAngs[0, i]
        s = allScales[0, i]
        x = allX[0, i]
        y = allY[0, i]

        # Find bin that is closet to a, s, x, y
        temp = abs(a - aBin)
        ia = 0
        for i in range(len(temp)):
            if temp[i] < temp[ia]:
                ia = i

        temp = abs(s - sBin)
        iS = 0
        for i in range(len(temp)):
            if temp[i] < temp[iS]:
                iS = i

        temp = abs(x - xBin)
        ix = 0
        for i in range(len(temp)):
            if temp[i] < temp[ix]:
                ix = i

        temp = abs(y - yBin)
        iy = 0
        for i in range(len(temp)):
            if temp[i] < temp[iy]:
                iy = i

        H[ia, iS, ix, iy] += 1

    # Find all bins with 3 or more features
    Bin_index = []
    l1, l2, l3, l4 = H.shape
    for i in range(l1):
        for j in range(l2):
            for k in range(l3):
                for l in range(l4):
                    if H[i, j, k, l] >= 3:
                        Bin_index.append((i, j, k, l))

    print('Peaks in the Hough array:')
    for i in range(len(Bin_index)):
        print('%d: %d points, (a, s, x, y) = %f, %f, %f, %f'
              % (i + 1, H[Bin_index[i][0], Bin_index[i][1], Bin_index[i][2], Bin_index[i][3]],
                 aBin[Bin_index[i][0]], sBin[Bin_index[i][1]],
                 xBin[Bin_index[i][2]], yBin[Bin_index[i][3]]))

    # Get the features corresponding to the largest bin
    nFeatures = np.max(H)
    print('Largest bin contains %d features' % nFeatures)
    Bin_index = []
    for i in range(l1):
        for j in range(l2):
            for k in range(l3):
                for l in range(l4):
                    if H[i, j, k, l] == nFeatures:
                        Bin_index.append((i, j, k, l))

    indices = []
    for idx in range(initial_len):
        a = allAngs[0, idx]
        s = allScales[0, idx]
        x = allX[0, idx]
        y = allY[0, idx]

        # Find bin that is closest to a, s, x, y
        temp = abs(a - aBin)
        ia = 0
        for i in range(len(temp)):
            if temp[i] < temp[ia]:
                ia = i

        temp = abs(s - sBin)
        iS = 0
        for i in range(len(temp)):
            if temp[i] < temp[iS]:
                iS = i

        temp = abs(x - xBin)
        ix = 0
        for i in range(len(temp)):
            if temp[i] < temp[ix]:
                ix = i

        temp = abs(y - yBin)
        iy = 0
        for i in range(len(temp)):
            if temp[i] < temp[iy]:
                iy = i

        if ia == Bin_index[0][0] and iS == Bin_index[0][1] and ix == Bin_index[0][2] and iy == Bin_index[0][3]:
            indices.append(idx)

    print('Features belonging to highest peak:')
    print(indices)
    return indices

def draw_matches(loc1, loc2, indices, path, QR, origin):
    I1 = cv2.imread(ROOT + QR)
    I2 = cv2.imread(ROOT + origin)

    h1, w1, temp = I1.shape
    h2, w2, temp = I2.shape
    matches_img = np.zeros((h1, w1 + w2, 3), np.uint8)
    matches_img[: h1, : w1] = I1
    matches_img[: h2, w1 : w1 + w2] = I2

    for idx in indices:
        (x1, y1) = (get_correspond_int(loc1[idx][1]), get_correspond_int(loc1[idx][0]))
        (x2, y2) = (get_correspond_int(loc2[idx][1]) + w1, get_correspond_int(loc2[idx][0]))
        cv2.line(matches_img, (x1, y1), (x2, y2), (255, 0, 0))
        cv2.putText(matches_img, str(idx + 1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))
        cv2.putText(matches_img, str(idx + 1), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255))

    # display(path, matches_img)
    cv2.imwrite(path, matches_img)

def mysiftalignment(I1, feature1, I2, feature2, out_path, QR, origin):
    '''
        The function aligns two images by using the SIFT features.
        Args:
            I1: The first image to be matched.
            feature1: Features of I1 extracted by SIFT.
            I2: The second image to be matched.
            feature2: Features of I2 extracted by SIFT.
            [out_path, QR, origin]: path for display and storage.
        Returns:
            len(indices): The number of matches pairs.
    '''
    des1, loc1 = feature1
    des2, loc2 = feature2

    matches = match(des1, des2)

    matches = np.array(matches).T
    des1_match = des1[matches[0, :], :]
    loc1_match = loc1[matches[0, :], :]
    des2_match = des2[matches[1, :], :]
    loc2_match = loc2[matches[1, :], :]

    indices = screenmatches(I1, I2, matches, loc1_match, des1_match, loc2_match, des2_match)

    indices = np.array(indices)
    draw_matches(loc1_match, loc2_match, indices, out_path, QR, origin)

    return len(indices)

def findBestMatching(I, I1, I2, I3):
    num = np.zeros((1, 3))
    source_QR = 'QR-Code.png'
    source_file1 = 'image1.png'
    source_file2 = 'image2.png'
    source_file3 = 'image3.png'
    output_file1 = '05QR_img1.png'
    output_file2 = '06QR_img2.png'
    output_file3 = '07QR_img3.png'

    des0, loc0 = sift(I)
    des1, loc1 = sift(I1)
    des2, loc2 = sift(I2)
    des3, loc3 = sift(I3)

    num[0, 0] = mysiftalignment(I1, [des0, loc0], I2, [des1, loc1], output_file1, source_QR, source_file1)
    num[0, 1] = mysiftalignment(I1, [des0, loc0], I2, [des2, loc2], output_file2, source_QR, source_file2)
    num[0, 2] = mysiftalignment(I1, [des0, loc0], I2, [des3, loc3], output_file3, source_QR, source_file3)

    index = 0
    for i in range(3):
        if num[0, i] > num[0, index]:
            index = i

    return index + 1

def Task1(FILENAME):
    '''
        Task1
        Read an image specified by FILENEMA and save it as
        '01original.jpg' in the current directory.
    '''
    output_file = '01original.jpg'

    Im = cv2.imread(ROOT + FILENAME, cv2.IMREAD_UNCHANGED)
    # display(output_file, Im)
    cv2.imwrite(OUT_PATH + output_file, Im)
    return Im

def Task2(Im):
    '''
        Task2
        Compute the corresponding binary edge image for the original image
        and save it as '02binary.jpg' in the current directory.
        Using Prewitt operator and given threshold T = (max intensity) * 0.2.
    '''
    output_file = '02binary1.jpg'

    threshold = np.max(Im) * 0.2
    g = myprewittedge(Im, threshold, "all")
    # display(output_file, g)
    cv2.imwrite(OUT_PATH + output_file, g)
    return g

def Task3(Im):
    '''
        Task3
        Compute the corresponding binary edge image for the original image
        and save it as '03binary.jpg' in the current directory.
        Using Prewitt operator and automatic computed threshold.
    '''
    output_file = '03binary2.jpg'

    f = myprewittedge(Im)
    # display(output_file, f)
    cv2.imwrite(OUT_PATH + output_file, f)
    return f

def Task4(f, FILENAME):
    '''
        Task4
        Find the longest segment extraction in the binary edge image
        generated by task3. Then, draw the line on the original image
        and save it as '04longestline.jpg' in the current directoy.
        Using skimage.transform package to perform Hough transform.
    '''
    output_file = '04longestline.jpg'

    max_sp, max_ep = mylineextraction(f)

    Im = cv2.imread(ROOT + FILENAME)

    cv2.circle(Im, (max_sp[1], max_sp[0]), 4, (0, 0, 255), 2)
    cv2.circle(Im, (max_ep[1], max_ep[0]), 4, (0, 0, 255), 2)
    cv2.line(Im, (max_sp[1], max_sp[0]), (max_ep[1], max_ep[0]), (255, 0, 0), 2)
    # display(output_file, Im)
    cv2.imwrite(OUT_PATH + output_file, Im)

def Task5():
    '''
        Task5(Image alignment using SIFT)
        For the three given images ('image1.png', 'image2.png', 'image3.png'), using SIFT
        to find out the one that matches the QR code image ('QR-Code.png') best. Then draw
        the matchings between the images and QR code image and save them as '05QR_img1.png',
        '06QR_img2.png', '07QR_img3.png' respectively.
    '''
    I = cv2.imread(ROOT + 'QR-Code.png', cv2.IMREAD_UNCHANGED)
    I1 = cv2.imread(ROOT + 'image1.png', cv2.IMREAD_UNCHANGED)
    I2 = cv2.imread(ROOT + 'image2.png', cv2.IMREAD_UNCHANGED)
    I3 = cv2.imread(ROOT + 'image3.png', cv2.IMREAD_UNCHANGED)

    n = findBestMatching(I, I1, I2, I3)
    print('The image matches QR-code.png best is image %d.png' % n)

def main():
    FILENAME = 'fig.tif'

    if len(sys.argv) == 2:
        FILENAME = sys.argv[1]

    Im = Task1(FILENAME)
    print('Original image is read and displayed successfully.')

    Im = im2double(Im)

    g = Task2(Im)
    print('The corresponding binary edge image is computed and dispalyed successfully.')
    
    f = Task3(Im)
    print('The corresponding binary edge image is computed and displayed successfully.')

    Task4(f, FILENAME)

    # Task5()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()