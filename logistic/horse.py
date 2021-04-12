"""
马是否存活 2020-04-26 19:35:37
"""
import numpy as np
from log_regres import sigmoid, stoc_grad_ascent


def classify(x, weights):
    y_ = sigmoid(np.dot(x, weights))
    if y_ > 0.5:
        return 1
    return 0


def test_horse():
    """
    测试马
    :return:
    """
    x_train = []
    y_train = []

    with open('horseColicTraining.txt') as f:
        lines = f.readlines()

    for line in lines:
        content = line.strip().split('\t')
        lin_arr = []
        for i in range(21):
            lin_arr.append(float(content[i]))
        x_train.append(lin_arr)
        y_train.append(float(content[21]))

    weights = stoc_grad_ascent(x_train, y_train, 500)

    with open('horseColicTest.txt') as f:
        lines = f.readlines()

    err_cnt = 0
    num_test = 0
    for line in lines:
        content = line.strip().split('\t')
        lin_arr = []
        for i in range(21):
            lin_arr.append(float(content[i]))
        if int(classify(np.array(lin_arr), weights)) != int(content[21]):
            err_cnt += 1
        num_test += 1

    err_rate = float(err_cnt / num_test)
    print(err_rate)
    return err_rate


def multi_test():
    num_test = 10
    err_sum = 0
    for i in range(num_test):
        err_sum += test_horse()

    print("average error rate is:{}".format(err_sum/num_test))
