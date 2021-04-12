"""kNN算法实现与应用 2020-04-14 14:38:13"""
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


def create_data():
    """创建数据"""
    # 数据集
    group = np.array([
                        [1.0, 1.1],
                        [1.0, 1.0],
                        [0, 0],
                        [0, 0.1]
    ])

    # 标签集
    labels = ['A', 'A', 'B', 'B']

    return group, labels


def f2matrix(file):
    """
    文本文件转numpy矩阵
    In:文本文件
    Out:训练样本矩阵和类标签向量
    """

    with open(file) as f:
        # 读入每一行
        lines = f.readlines()

    m = len(lines)
    # 创建全0矩阵及标签向量
    matrix = np.zeros((m, 3))   # 三个特征
    labels = []

    # 写入矩阵与标签
    for idx, line in enumerate(lines):
        line = line.strip().split('\t')
        matrix[idx] = line[:3]
        labels.append(line[-1])

    return matrix, np.array(labels)


def data_norm(data):
    """归一化数据到0-1"""
    # (当前数据-每个特征的最小值)/(每个特征的最大值-每个特征的最小值)
    max_vals = data.max(axis=0)
    min_vals = data.min(axis=0)
    data = (data - min_vals)/(max_vals - min_vals)
    return data


def plot_data(x1, x2, labels):
    """
    可视化数据集
    In: x1,x2:数据集的2个特征
        labels:标签集
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, x2, 15.0*np.array(labels), 15.0*np.array(labels))


def classify0(x, data, labels, k):
    """
    kNN算法实现
    In: x:待分类的输入向量 1×n n为特征数
        data:数据集    m×n
        labels:标签集
        k:kNN参数k
    Out: x的预测分类
    """
    # 计算x与数据集每个向量的欧式距离
    distance = np.sum((x - data)**2, axis=1)

    # 将数据集索引按照距离排序(升序)
    dist_idx = distance.argsort()

    # 计算距离最小前k个数据各标签的个数
    class_count = {}
    for idx in dist_idx[:k]:
        label = labels[idx]
        class_count[label] = class_count.get(label, 0) + 1

    # 排序，选择标签个数最多的作为分类结果
    res = Counter(class_count).most_common()[0][0]

    return res


def class_test(file, k=3, ratio=0.1):
    """测试错误率"""
    # 读入文本文件
    data, labels = f2matrix(file)
    m = labels.shape[0]         # 获取样本总数
    test_num = int(m*ratio)     # 计算测试集样本总数

    # 归一化数据
    norm_data = data_norm(data)

    # 测试
    error_cnt = 0
    for i in range(test_num):
        # 数据集中前test_num个数据作为测试集
        predict = classify0(norm_data[i], norm_data[test_num:], labels[test_num:], k)
        print("predict:{}   real:{}\n".format(predict, labels[i]))
        if predict != labels[i]:
            error_cnt += 1
    print("测试集总数为{}，错误个数为{}，错误率为{}%\n".format(test_num, error_cnt, error_cnt/float(test_num)*100))

