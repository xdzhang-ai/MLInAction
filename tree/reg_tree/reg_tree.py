"""
树回归算法的实现与应用 2020-05-13 13:41:13
"""
import numpy as np


def load_data(file):
    data = []
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split('\t')
        data.append(np.array(line, dtype='float'))

    return np.array(data)


def bi_split_data(data, feat, value):
    """
    将数据根据对应的特征值二分
    :param data: 数据集
    :param feat: 特征
    :param value: 特征值
    :return: 二分后的数据
    """
    left = data[data[:, feat] > value]
    right = data[data[:, feat] <= value]

    return left, right


def reg_leaf(data):
    """
    生成叶结点的函数（取均值）
    :param data:
    :return:
    """
    return np.mean(data[:, -1])


def reg_err(data):
    """
    在选择最佳划分特征时用来衡量划分质量的误差函数
    :param data:
    :return:
    """
    return np.var(data[:, -1]) * data.shape[0]


def linear_tree(data):
    """
    模型树叶结点生成函数
    :param data:
    :return:
    """
    m, n = data.shape
    X = np.ones((m, n))
    X[:, 1: n] = data[:, 0: n-1]
    Y = data[:, -1]
    xtx = X.T @ X
    if np.linalg.det(xtx) == 0:
        raise NameError('矩阵不可逆，尝试提升ops的第二个值')
    w = np.linalg.inv(xtx) @ X.T @ Y
    return w, X, Y


def model_leaf(data):
    """
    生成模型树叶结点函数
    :param data:
    :return:
    """
    w, X, Y = linear_tree(data)
    return w


def model_err(data):
    """
    模型树误差计算函数
    :param data:
    :return:
    """
    w, X, Y = linear_tree(data)
    y_ = w @ X.T
    return np.sum((y_ - Y)**2)


def get_best_feat(data, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    选择最佳划分特征
    :param data:
    :param leaf_type: 建立叶结点的函数
    :param err_type: 误差计算函数
    :param ops: 构建树所需参数
    :return: 如果满足停止条件则返回叶结点，否则返回划分特征及划分值
    """
    tol_e = ops[0]  # 容许的误差下降值
    tol_n = ops[1]  # 切分的最少样本数
    # 当数据集只有一个样本时返回none和叶子结点值
    if len(data[:, -1]) == 1:
        return None, leaf_type(data)
    m, n = data.shape

    # 计算未划分误差及划分后误差并比较
    err = err_type(data)
    best_err = np.inf   # 初始化划分后的误差
    best_idx = 0    # 初始化最佳特征
    best_value = 0  # 初始化最佳特征划分值
    # 遍历特征寻找使误差下降最大的划分特征及划分值
    for feat in range(n-1):     # 最后一列为标签
        for split_val in set(data[:, feat]):
            left, right = bi_split_data(data, feat, split_val)  # 二分数据
            if left.shape[0] < tol_n or right.shape[0] < tol_n:
                continue
            new_err = err_type(left) + err_type(right)  # 划分后的误差值

            if new_err < best_err:
                best_err = new_err
                best_idx = feat
                best_value = split_val

    # 若误差下降很小时则返回叶结点
    if (err - best_err) < tol_e:
        return None, leaf_type(data)

    left, right = bi_split_data(data, best_idx, best_value)
    # 若划分后的左右数据集样本数有一个比最小要求的样本数小则返回叶结点
    if left.shape[0] < tol_n or right.shape[0] < tol_n:
        return None, leaf_type(data)

    return best_idx, best_value


def create_tree(data, leaf_type=reg_leaf, err_type=reg_err, ops=(1, 4)):
    """
    创建树
    :param data:
    :param leaf_type:
    :param err_type:
    :param ops:
    :return:
    """
    feat, val = get_best_feat(data, leaf_type, err_type, ops)
    # 判断是否是叶结点
    if feat is None:
        return val

    tree = {}
    tree['feat'] = feat
    tree['val'] = val

    left, right = bi_split_data(data, feat, val)

    tree['left'] = create_tree(left, leaf_type, err_type, ops)
    tree['right'] = create_tree(right, leaf_type, err_type, ops)

    return tree


def reg_tree_eval(leaf, data):
    """
    回归树预测函数
    :param leaf: 叶结点
    :param data:
    :return:
    """
    return float(leaf)


def model_tree_eval(leaf, data):
    """
    模型树预测函数
    :param leaf:
    :param data:
    :return:
    """
    if len(data.shape) > 1:
        n = data.shape[1]
    else:
        n = data.shape[0]

    x = np.ones(n)
    x[1: n] = data[: n-1]
    return leaf @ x.T


def predict_one(tree, data, model_eval=reg_tree_eval):
    """
    预测一个数据
    :param tree: 生成的树
    :param data: 待预测数据
    :param model_eval: 预测函数
    :return:
    """
    # 判断是否为叶结点
    if not is_tree(tree):
        return model_eval(tree, data)
    if data[tree['feat']] > tree['val']:
        if not is_tree(tree['left']):
            return model_eval(tree['left'], data)
        return predict_one(tree['left'], data, model_eval)
    else:
        if not is_tree(tree['right']):
            return model_eval(tree['right'], data)
        return predict_one(tree['right'], data, model_eval)


def predict(tree, test_data, model_eval=reg_tree_eval):
    """
    预测一组数据
    :param tree: 生成的树
    :param test_data: 测试数据
    :param model_eval: 预测函数
    :return:
    """
    m = test_data.shape[0]
    y_s = np.zeros(m)
    for i in range(m):
        y_ = predict_one(tree, test_data[i], model_eval)
        y_s[i] = y_
    return y_s


"""
###############
后剪枝
###############
"""


def is_tree(obj):
    # 判断是否是一棵树（字典）
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    从上往下遍历树直到叶节点，返回叶结点的平均值
    :param tree:
    :return:
    """
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])

    return (tree['left'] + tree['right']) / 2


def prune(tree, test_data):
    """
    后剪枝
    :param tree:
    :param test_data:
    :return:
    """
    if test_data.shape[0] == 0:
        return get_mean(tree)

    if is_tree(tree['right']) or is_tree(tree['left']):
        lSet, rSet = bi_split_data(test_data, tree['feat'], tree['val'])

    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)

    # 合并
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        lSet, rSet = bi_split_data(test_data, tree['feat'], tree['val'])
        errorNoMerge = sum(np.power(lSet[:, -1] - tree['left'],2)) + sum(np.power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(np.power(test_data[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge:
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree



np.concatenate
