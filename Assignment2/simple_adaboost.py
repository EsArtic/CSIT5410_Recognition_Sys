import math

def normalize(w):
    total = 0
    ret = []
    for elem in w:
        total += elem

    for elem in w:
        ret.append(elem / total)

    return ret

def error(weight, y, h):
    ret = 0
    for i in range(len(weight)):
        ret += weight[i] * (1 - h[i] * y[i]) / 2

    return ret

def update(w, alpha, y, h):
    ret = []
    for i in range(len(w)):
        temp = w[i] * math.pow(math.e, -1 * alpha * y[i] * h[i])
        ret.append(temp)

    return ret

def main():
    weight = [1/9] * 9
    y = [1, 1, 1, 1, -1, -1, -1, -1, -1]
    h1 = [1, 1, 1, 1, -1, -1, -1, 1, -1]
    h2 = [1, 1, -1, 1, 1, -1, -1, 1, -1]
    h3 = [-1, 1, 1, 1, -1, -1, 1, -1, 1]
    weak_classifiers = [h1, h2, h3]
    alpha = []
    classifers = []
    visited = [False] * len(weak_classifiers)

    for i in range(2):
        print('Iteration %d' % (i + 1))
        weight = normalize(weight)
        print('Normalized weights:', weight)
        selected = 0
        min_error = 10000
        for j in range(len(weak_classifiers)):
            if not visited[j]:
                ej = error(weight, y, weak_classifiers[j])
                if ej < min_error:
                    selected = j
                    min_error = ej
                print('Error%d: %f' % (j, ej))
        classifers.append(selected)
        visited[selected] = True
        curr_alpha = 0.5 * math.log((1 - min_error) / min_error)
        alpha.append(curr_alpha)
        print('Pick classifer %d, weight: %f' % (selected, curr_alpha))

        weight = update(weight, curr_alpha, y, weak_classifiers[selected])
        print('Updated weights:', weight)
        print()

    print()
    print('Strong classifier:', classifers)
    print('Classifer weights:', alpha)
    print('Data weights:     ', weight)
    print()
    print('Test:')
    print('y =', y)
    predict = []
    for i in range(len(weight)):
        total = 0
        for j in range(len(classifers)):
            total += alpha[j] * weak_classifiers[classifers[j]][i]

        if total > 0:
            predict.append(1)
        else:
            predict.append(-1)
    print('predict =', predict)
    print('Error =', error(weight, y, predict))

if __name__ == '__main__':
    main()