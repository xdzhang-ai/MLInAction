"""
逻辑回归算法实现与应用 2020-04-26 16:38:27
"""
import numpy as np
import random


def load_data():
    """
    加载数据集
    :return:    数据集和标签集
    """
    data = []
    labels = []
    with open('testSet.txt') as f:
        lines = f.readlines()

    for line in lines:
        content = line.strip().split()
        data.append([1.0, float(content[0]), float(content[1])])
        labels.append(int(content[-1]))

    return data, labels


def sigmoid(x):
    """
    sigmoid激活函数
    :param x: 输入
    :return: 输出
    """
    if x > 0:
        return 1.0/(1.0 + np.exp(-x))
    else:
        return np.exp(x) / (np.exp(x) + 1.0)


def grad_ascent(data, labels):
    """
    梯度上升迭代
    :param data:数据集
    :param labels: 标签集
    :return: 参数
    """

    data = np.array(data)   # m×n
    labels = np.array(labels)   # (m,)
    y = labels[:, np.newaxis]   # (m, 1)
    # 获取样本数m和数据特征数n
    m, n = data.shape

    # 初始化参数
    weights = np.ones((n, 1))     # 初始化权重为n×1的全1向量
    alpha = 0.001       # learning_rate
    iteration = 500     # 迭代次数

    for i in range(iteration):
        y_ = sigmoid(np.dot(data, weights))
        error = y - y_      # m×1
        weights += alpha * np.dot(data.T, error)
    return weights


def stoc_grad_ascent(data, labels, num_iter=150):
    """
    随机梯度上升
    :param data: 数据集
    :param labels:  标签集
    :param num_iter: 迭代次数
    :return: 权重参数
    """
    data = np.array(data)
    m, n = data.shape
    weights = np.ones(n)

    # 循环迭代
    for j in range(num_iter):
        data_idx = list(range(m))
        # 遍历所有样本
        for i in range(m):
            alpha = 4/(1 + i + j) + 0.01    # 根据迭代次数更新alpha
            index = int(random.uniform(0, len(data_idx)))   # 随机选取一个样本
            y_ = sigmoid(np.dot(weights, data[index]))
            error = labels[index] - y_
            weights += alpha * error * data[index]
            del(data_idx[index])    # 删除已选样本
    return weights


def plot_bound(weights):
    """
    画出决策边界
    :param weights: 训练出的权重
    :return: 无
    """
    import matplotlib.pyplot as plt
    data, labels = load_data()
    data = np.array(data)
    m = data.shape[0]

    x0 = [data[i][1] for i in range(m) if labels[i] == 0]
    y0 = [data[i][2] for i in range(m) if labels[i] == 0]
    x1 = [data[i][1] for i in range(m) if labels[i] == 1]
    y1 = [data[i][2] for i in range(m) if labels[i] == 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x0, y0, s=30, c='red', marker='s')
    ax.scatter(x1, y1, s=30, c='green')

    x = np.arange(-3, 3, 0.1)
    y = (-weights[0] - weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

