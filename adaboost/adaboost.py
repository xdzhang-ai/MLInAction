"""
adaboost算法实现 2020-05-02 16:39:27
"""
import numpy as np


def create_data():
    """创建数据"""
    # 数据集
    group = np.array([
                        [1.0, 2.1],
                        [2.0, 1.1],
                        [1.3, 1.0],
                        [1.0, 1.0],
                        [2.0, 1.0]
    ])

    # 标签集
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]

    return group, labels


"""
### 构建弱分类器：单层决策树
"""


def classify_by_feat(data, feat, threshold, inequal):
    """
    将数据集根据指定特征按照给定阈值分类
    :param data: 数据集
    :param feat: 特征索引
    :param threshold: 阈值
    :param inequal: 不等式
    :return: 预测分类向量
    """
    m, n = data.shape
    predicts = np.ones(m)
    if inequal == 'lt':
        predicts[data[:, feat] <= threshold] = -1
    else:
        predicts[data[:, feat] > threshold] = -1

    return predicts


def build_tree(data, labels, D):
    """
    构造单层决策树
    :param data:数据集
    :param labels: 标签集
    :param D: 样本分布权重  m×1
    :return: 单层决策树信息
    """
    err_min = np.inf    # 初始化最小错误率
    tree = {}   # 树信息
    m, n = data.shape
    step_num = 10   # 特征值遍历步数

    # 第一层遍历特征
    for feat in range(n):
        min_val = min(data[:, feat])
        max_val = max(data[:, feat])
        # 计算步长
        step_size = (max_val - min_val) / step_num
        for i in range(-1, step_num+1):
            for inequal in ['lt', 'gt']:
                # 将遍历到的值设为阈值
                thr = min_val + step_size * i
                # 预测分类
                predicts = classify_by_feat(data, feat, thr, inequal)
                # 计算错误率
                err_vector = np.zeros(m)
                err_vector[predicts != labels] = 1
                weighted_error = err_vector.T @ D   # 计算加权错误率

                # 若错误率小于最小错误率则更新
                if weighted_error < err_min:
                    err_min = weighted_error
                    best_predict = predicts

                    tree['feat'] = feat
                    tree['thresh'] = thr
                    tree['ineq'] = inequal

    return tree, err_min, best_predict


def adaboost_train(data, labels, num_iter=40):
    """
    基于单层决策树弱分类器的adaboost训练算法
    :param data: 数据集
    :param labels: 标签集
    :param num_iter: 迭代数
    :return: 弱分类器列表
    """
    # 构造弱分类器列表
    weak_class = []
    m = data.shape[0]
    class_est = np.zeros(m)     # 类别估计累计值
    D = np.ones(m) / m  # 数据权重向量

    for i in range(num_iter):
        tree, error, predict = build_tree(data, labels, D)
        print("D:{}".format(D))

        # 每个弱分类器的权重
        alpha = 0.5 * np.log((1.0-error) / max(error, 1e-16))
        tree['alpha'] = alpha
        weak_class.append(tree)

        # 更新样本权重D
        D *= np.exp(-(predict * labels)*alpha)
        D /= D.sum()
        # D[predict == labels] *= np.exp(-alpha) / D_sum
        # D[predict != labels] *= np.exp(alpha) / D_sum

        class_est += alpha * predict    # 更新累计估计值

        # 根据累计估计值计算当前错误率
        error_rate = (np.sign(class_est) != labels).sum() / m
        print("error rate:{}".format(error_rate))

        if error_rate == 0:
            break
    return weak_class, class_est


def ada_classify(data, weak_class):
    """
    预测分类
    :param data: 待分类数据
    :param weak_class: 弱分类器列表
    :return: 预测向量
    """
    data = np.array(data)
    if len(data.shape) == 1:
        data = data[np.newaxis, :]
    agg_predicts = np.zeros(data.shape[0])
    for i in range(len(weak_class)):
        tree = weak_class[i]
        predicts = classify_by_feat(data, tree['feat'], tree['thresh'], tree['ineq'])
        agg_predicts += tree['alpha'] * predicts
    return np.sign(agg_predicts)


def plot_roc(class_est, labels):
    """
    绘制ROC并计算AUC
    :param class_est: 连续型估计值
    :param labels: 标签集
    :return: 绘制ROC、计算AUC
    """
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)  # cursor
    ySum = 0.0  # variable to calculate AUC
    numPosClas = sum(np.array(labels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(labels) - numPosClas)
    sortedIndicies = class_est.argsort()  # get sorted index, it's reverse

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndicies:
        if labels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum * xStep)
