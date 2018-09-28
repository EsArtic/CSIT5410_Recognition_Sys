import cv2
import numpy as np

def im2double(img):
    result = np.zeros(img.shape, np.float64)
    height, width = result.shape
    for i in range(height):
        for j in range(width):
            result[i, j] = img[i, j] / 255.

    return result

def main():
    Im = cv2.imread('./fig.tif', cv2.IMREAD_UNCHANGED)
    Im = im2double(Im)

    maxx = np.max(Im)
    minn = np.min(Im)
    print(Im)
    print(maxx, minn)

if __name__ == '__main__':
    main()