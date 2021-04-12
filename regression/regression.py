"""
线性回归算法实现与应用 2020-05-09 13:57:49
"""
import numpy as np


def load_data(file):
    """
    加载数据
    :param file: 文件名
    :return:
    """
    with open(file) as f:
        num_feat = len(f.readline().strip().split('\t')) - 1    # 计算特征数

    with open(file) as f:
        lines = f.readlines()

    data = []
    labels = []
    for line in lines:
        line = line.strip().split('\t')
        x = []
        for i in range(num_feat):
            x.append(float(line[i]))
        data.append(x)
        labels.append(float(line[-1]))

    return np.array(data), np.array(labels)


def form_regres(data, labels):
    """
    利用公式w=(X.TX)^-1*(X.Ty)计算回归系数
    :param data:
    :param labels:
    :return: 回归系数向量w
    """
    xtx = data.T @ data
    # 判断xtx是否可逆
    if np.linalg.det(xtx) == 0:
        print("矩阵不可逆")
        return -1
    w = np.linalg.inv(xtx) @ (data.T @ labels)
    return w


def lwlr(test_point, data, labels, k=1.0):
    """
    局部加权线性回归    w=(X.TWX)^-1X.TWy
    :param test_point: 待预测点
    :param data:
    :param labels:
    :param k: 高斯核均方
    :return: 回归系数向量w
    """

    # 计算加权矩阵W(m×m，只在对角线上有元素，每个样本附近的点都有较高的权重)
    m = data.shape[0]
    W = np.zeros((m, m))
    for i in range(m):
        delta = test_point - data[i]
        W[i][i] = np.exp(-0.5 * delta @ delta.T / (k**2))
    xtwx = data.T @ W @ data

    if np.linalg.det(xtwx) == 0:
        print('矩阵不可逆')
        return -1
    w = np.linalg.inv(xtwx) @ data.T @ W @ labels
    return w


def lwlr_test(test_arr, data, labels, k=1.0):
    m = test_arr.shape[0]
    y_s = np.zeros(m)
    for i in range(m):
        test_point = test_arr[i]
        y_s[i] = lwlr(test_point, data, labels, k) @ test_point

    return y_s


def plot_res(data, labels, y_s):
    """
    画出结果
    :param data:
    :param labels:
    :param y_s:
    :return:
    """
    import matplotlib.pyplot as plt
    figure = plt.figure()
    ax = figure.add_subplot(111)
    ax.scatter(data[:, 1], labels, s=2, c='red')
    # 排序
    index = data[:, 1].argsort(0)
    ax.plot(data[index][:, 1], y_s[index])
    plt.show()