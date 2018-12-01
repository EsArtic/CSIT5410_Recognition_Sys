import math
import numpy as np

def LBP(region):
    LBP_represent = np.zeros(region.shape, dtype = int)
    value = region[1, 1]
    n, m = region.shape
    for i in range(n):
        for j in range(m):
            if region[i, j] >= value:
                LBP_represent[i, j] = 1

    return LBP_represent

def get_binary(LBP_represent):
    binary = ''
    binary += str(int(LBP_represent[2, 2]))
    binary += str(int(LBP_represent[2, 1]))
    binary += str(int(LBP_represent[2, 0]))
    binary += str(int(LBP_represent[1, 0]))
    binary += str(int(LBP_represent[0, 0]))
    binary += str(int(LBP_represent[0, 1]))
    binary += str(int(LBP_represent[0, 2]))
    binary += str(int(LBP_represent[1, 2]))

    return binary

def get_decimal(binary):
    decimal = 0
    index = len(binary) - 1
    for i in range(len(binary)):
        curr = int(int(binary[index]) * math.pow(2, i))
        # print(curr, end = ' ')
        decimal += curr
        index -= 1

    return decimal

def rotate(region):
    ret = np.zeros(region.shape, dtype = int)
    ret[1, 1] = region[1, 1]
    ret[0, 0] = region[1, 0]
    ret[0, 1] = region[0, 0]
    ret[0, 2] = region[0, 1]
    ret[1, 2] = region[0, 2]
    ret[2, 2] = region[1, 2]
    ret[2, 1] = region[2, 2]
    ret[2, 0] = region[2, 1]
    ret[1, 0] = region[2, 0]

    return ret

def main():
    initial = np.array([[34, 67, 36],
                        [25, 67, 72],
                        [75, 96, 35]], dtype = int)
    print('Initial region = ')
    print(initial)
    print()

    LBP_represent = LBP(initial)
    print("LBP representation for the region = ")
    print(LBP_represent)
    print()

    print("Binary value = ")
    binary = get_binary(LBP_represent)
    print(binary)
    print()

    print("Decimal value = ")
    decimal = get_decimal(binary)
    print(decimal)
    print()

    min_value = 255
    min_LBP = None
    min_region = None
    min_binary = None
    curr = initial.copy()
    for i in range(7):
        curr = rotate(curr)
        '''
        print('Roated region %d = ' % (i + 1))
        print(curr)
        print()
        '''
        LBP_represent = LBP(curr)
        '''
        print(LBP_represent)
        print()
        print("Binary value = ")
        '''
        binary = get_binary(LBP_represent)
        '''
        print(binary)
        print()
        print("Decimal value = ")
        '''
        decimal = get_decimal(binary)
        '''
        print(decimal)
        print()
        '''
        if decimal < min_value:
            min_value = decimal
            min_LBP = LBP_represent
            min_region = curr
            min_binary = binary

    print('Roateted min region = ')
    print(min_region)
    print()

    print("LBP representation for the region = ")
    print(min_LBP)
    print()

    print("Binary value = ")
    print(min_binary)
    print()

    print("Decimal value = ")
    print(min_value)
    print()

    '''
        Rotate by 180 degree:
        [[34, 67, 36],         [[35, 96, 75],
         [25, 67, 72],  ====>   [72, 67, 25],
         [75, 96, 35]]          [36, 67, 34]]
    '''
    rotated = np.array([[35, 96, 75],
                        [72, 67, 25],
                        [36, 67, 34]], dtype = int)
    print('Rotated region = ')
    print(rotated)
    print()

    LBP_represent = LBP(rotated)
    print("LBP representation for the region = ")
    print(LBP_represent)
    print()

    print("Binary value = ")
    binary = get_binary(LBP_represent)
    print(binary)
    print()

    print("Decimal value = ")
    decimal = get_decimal(binary)
    print(decimal)
    print()

    min_value = 255
    min_LBP = None
    min_region = None
    min_binary = None
    curr = rotated.copy()
    for i in range(7):
        curr = rotate(curr)
        '''
        print('Roated region %d = ' % (i + 1))
        print(curr)
        print()
        '''
        LBP_represent = LBP(curr)
        '''
        print(LBP_represent)
        print()
        print("Binary value = ")
        '''
        binary = get_binary(LBP_represent)
        '''
        print(binary)
        print()
        print("Decimal value = ")
        '''
        decimal = get_decimal(binary)
        '''
        print(decimal)
        print()
        '''
        if decimal < min_value:
            min_value = decimal
            min_LBP = LBP_represent
            min_region = curr
            min_binary = binary

    print('Roateted min region = ')
    print(min_region)
    print()

    print("LBP representation for the region = ")
    print(min_LBP)
    print()

    print("Binary value = ")
    print(min_binary)
    print()

    print("Decimal value = ")
    print(min_value)
    print()

    '''
        Part 3
    '''
    changed = np.array([[34, 70, 38],
                        [28, 69, 71],
                        [76, 95, 34]], dtype = int)
    print('Rotated region = ')
    print(changed)
    print()

    LBP_represent = LBP(changed)
    print("LBP representation for the region = ")
    print(LBP_represent)
    print()

    print("Binary value = ")
    binary = get_binary(LBP_represent)
    print(binary)
    print()

    print("Decimal value = ")
    decimal = get_decimal(binary)
    print(decimal)
    print()

    min_value = 255
    min_LBP = None
    min_region = None
    min_binary = None
    curr = changed.copy()
    for i in range(7):
        curr = rotate(curr)
        '''
        print('Roated region %d = ' % (i + 1))
        print(curr)
        print()
        '''
        LBP_represent = LBP(curr)
        '''
        print(LBP_represent)
        print()
        print("Binary value = ")
        '''
        binary = get_binary(LBP_represent)
        '''
        print(binary)
        print()
        print("Decimal value = ")
        '''
        decimal = get_decimal(binary)
        '''
        print(decimal)
        print()
        '''
        if decimal < min_value:
            min_value = decimal
            min_LBP = LBP_represent
            min_region = curr
            min_binary = binary

    print('Roateted min region = ')
    print(min_region)
    print()

    print("LBP representation for the region = ")
    print(min_LBP)
    print()

    print("Binary value = ")
    print(min_binary)
    print()

    print("Decimal value = ")
    print(min_value)
    print()

if __name__ == '__main__':
    main()