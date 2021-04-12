"""
决策树算法实现与应用 2020-04-15 15:58:35
"""
from math import log
import numpy as np


def create_data():
    """
    创建数据集用于测试
    :return: 数据集
    """
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return np.array(data), labels


def cal_entropy(data):
    """
    计算数据集信息熵
    :param data: 数据集 m*3 每行最后一列为标签
    :return: 数据集的信息熵
    """
    # 样本数
    m = len(data)

    # 记录数据集中各类样本的个数，键为标签值，值为个数
    label_cnt = {}

    for d in data:
        label = d[-1]
        label_cnt[label] = label_cnt.get(label, 0) + 1

    # 计算熵
    entropy = 0
    for num_label in label_cnt.values():
        prob = num_label / m    # 各标签出现概率
        entropy -= prob * log(prob, 2)

    return entropy


def split_data(data, axis, value):
    """
    根据给定特征分割数据集
    :param data: 待分割数据集 m*n
    :param axis: 给定特征的索引
    :param value: 给定特征对应值
    :return: 数据子集
    """
    subset = []
    for d in data:
        if d[axis] == value:
            subset_data_left = d[:axis]
            subset_data_right = d[axis+1:]
            subset_data = np.concatenate((subset_data_left, subset_data_right))
            subset.append(subset_data)

    return np.array(subset)


def get_best_feature(data):
    """
    根据信息增益选择最好的特征（信息增益越大越好）
    :param data: 待划分数据集
    :return: 最好特征
    """
    # 初始化最好特征索引
    best_feature = -1

    # 获取特征数，最后一列是标签
    num_feature = len(data[0]) - 1

    # 原数据熵
    entropy = cal_entropy(data)

    # 初始化信息增益（老熵-新熵）
    best_info_gain = 0

    # 遍历特征寻找最好特征
    for feat_axis in range(num_feature):
        # 获取此特征上的所有取值
        values = set(data[:, feat_axis])
        new_entropy = 0

        # 遍历所有取值得到数据子集
        for value in values:
            # 获取此特征取此值时的数据子集
            subset = split_data(data, feat_axis, value)

            # 计算熵并累加
            prob = len(subset)/len(data)
            new_entropy += prob * cal_entropy(subset)

        info_gain = entropy - new_entropy
        # 基于此特征值计算出的当前信息增益大于最大信息增益时则更新
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feat_axis

    return best_feature


def vote_classs(class_list):
    """
    对于已经遍历完数据所有特征后的节点中仍出现标签不同的情况，采用投票法确定节点分类
    :param class_list: 节点中的类列表
    :return: 出现次数最高的类标签
    """
    class_dct = {}
    for cla in class_list:
        class_dct[cla] = class_dct.get(cla, 0) + 1

    # 对类排序
    sorted_class = sorted(class_dct.items(), key=lambda kv: (kv[1]))
    return sorted_class[-1][0]


def create_tree(data, labels):
    """
    递归创建树
    :param data:数据集，最后一行为标签
    :param labels: 特征标签列表，存储所有特征的标签(值)
    :return: 树结构
    """
    # 判断停止条件
    # 1 当前所有数据同属一类
    class_list = data[:, -1]    # 获取标签值
    if list(class_list).count(class_list[0]) == len(class_list):
        return class_list[0]
    # 2 当前已经遍历所有特征
    if len(class_list[0]) == 1:     # 只剩标签列
        return vote_classs(class_list)

    # 递归生成节点
    best_feat = get_best_feature(data)  # 选择最好特征索引
    feat_name = labels[best_feat]       # 根据特征索引获取特征名字

    # 初始化树
    my_tree = {feat_name: {}}

    del(labels[best_feat])
    feat_values = set(data[:, best_feat])
    for value in feat_values:
        sub_labels = labels[:]
        my_tree[feat_name][value] = create_tree(split_data(data, best_feat, value), sub_labels)

    return my_tree


def classify(tree, labels, x):
    """
    利用决策树分类，仍采用递归方法
    :param tree: 决策树（字典嵌套）
    :param labels: 特征名称列表
    :param x: 输入的待分类数据
    :return: x的类别
    """
    feat = list(tree.keys())[0]      # 获取特征名
    feat_idx = labels.index(feat)       # 获取特征名对应的索引
    for value in tree[feat]:
        if x[feat_idx] == type(x[feat_idx])(value):
            sub_tree = tree[feat][value]
            if type(sub_tree).__name__ == 'dict':
                return classify(sub_tree, labels, x)
            else:
                return sub_tree


# if __name__ == '__main__':
#     data, labels = create_data()
#     labels2 = labels[:]
#     mytree = create_tree(data, labels)
#     res = classify(mytree, labels2, [1, 1])