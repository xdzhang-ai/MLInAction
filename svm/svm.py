"""
svm算法实现与应用  2020-04-28 16:48:42
"""
import random
import numpy as np


def load_data(file, k_tup=('rbf', 2)):
    """
    加载数据
    :param file:文件名
    :return: 数据集和标签集
    """
    with open(file) as f:
        lines = f.readlines()

    data = []
    labels = []
    for line in lines:
        l = line.strip().split('\t')
        data.append([float(l[0]), float(l[1])])
        labels.append(float(l[2]))

    m = len(labels)
    K = np.zeros((m, m))
    for i in range(m):
        K[i] = create_kernel(data, data[i], k_tup)

    return data, labels, K


def select_j(i, m):
    """
    再SMO算法中选择一对α，此函数在给定一个αi的基础上，选择一个与αi不同的αj
    :param i: 给定α的下标
    :param m: α的总数
    :return: αj的下标
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))

    return j


def clip_alpha(a, H, L):
    """
    对数值过大或过小的α做调整
    :param a: α
    :param H: 上界
    :param L: 下界
    :return: 调整后的alpha
    """

    if a > H:
        return H
    elif a < L:
        return L
    else:
        return a


def smo_simple(data, labels, C, toler, max_iter):
    """
    smo算法简化版
    :param data: 数据集
    :param labels: 标签集
    :param C: 常数C
    :param toler: 容错率
    :param max_iter: 最大循环次数
    :return: b和alphas
    """
    # 数据预处理
    data = np.array(data)
    labels = np.array(labels)
    m, n = data.shape

    # 初始化b和alphas
    b = 0
    alphas = np.zeros(m)

    iter = 0    # 存储没有alpha改变的情况下遍历数据集的次数
    while iter < max_iter:
        a_changed = 0   # 记录alpha是否进行优化
        # 遍历样本
        for i in range(m):
            yi_ = float(np.dot(alphas * labels, np.dot(data, data[i]))) + b     # xi的预测值：(w.T)Xi+b
            ei = yi_ - float(labels[i])     # 计算与真实值误差，若较大则要对αi进行更改
            # 判断是否更改α
            if ((labels[i] * ei < -toler) and alphas[i] < C) or ((labels[i] * ei > toler) and alphas[i] > 0):
                j = select_j(i, m)  # 选取另一个α
                yj_ = float(np.dot(alphas * labels, np.dot(data, data[i]))) + b     # xj的预测值
                ej = yj_ - float(labels[j])

                ai_old = alphas[i].copy()   # 记录原来的αi的值
                aj_old = alphas[j].copy()   # 记录原来的αj的值

                # 计算L和H，保证alpha在0与C之间
                if labels[i] != labels[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] - alphas[i])

                # L=H则不改变
                if L == H:
                    print('L==H')
                    continue

                # 计算αj的最优修改量
                eta = 2.0 * np.dot(data[i], data[j]) - np.dot(data[j], data[j]) - np.dot(data[i], data[i])
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 对αj进行调整
                alphas[j] -= labels[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - aj_old) < 0.00001:
                    print('j not move enough')
                    continue
                # 对αi做同αj相反方向变化
                alphas[i] += labels[i] * labels[j] * (aj_old - alphas[j])

                # 设置常数项b
                b1 = b - ei - labels[i] * (alphas[i] - ai_old) * np.dot(data[i], data[i]) - labels[j] * (alphas[j] - aj_old) * np.dot(data[i], data[j])
                b2 = b - ej - labels[i] * (alphas[i] - ai_old) * np.dot(data[i], data[j]) - labels[j] * (alphas[j] - aj_old) * np.dot(data[j], data[j])
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                a_changed += 1
                print("iter:{} i:{}, pairs changed: {}".format(iter, i, a_changed))

        if a_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number:{}'.format(iter))

        return b, alphas


def calcWs(alphas,data,labels):
    data = np.array(data)
    labels = np.array(labels)
    m,n = np.shape(data)
    w = np.zeros(n)
    for i in range(m):
        w += alphas[i]*labels[i] *data[i]
    return w


def test(data, labels, ws, b):
    err_cnt = 0
    for i in range(len(labels)):
        if labels[i] * (data[i] @ ws + b) < 0:
            err_cnt += 1
    return err_cnt / len(labels)


"""
####################
核函数版本
####################
"""


def create_kernel(X, A, k_tup):
    """
    创建核函数
    :param X: 数据集data
    :param A: 数据集中的某个样本
    :param k_tup: 存放核函数信息的元组，第一个元素为核函数种类，第二个为核函数参数
    :return: 核化后的数据
    """
    X = np.array(X)
    m, n = np.shape(X)

    # 判断核函数种类
    if k_tup[0] == 'lin':   # 线性核
        K = X @ A.T
    elif k_tup[0] == 'rbf':     #高斯核
        K = np.zeros(m)
        for i in range(m):
            delta = X[i] - A
            K[i] = delta @ delta.T
        K = np.exp(-K/(k_tup[1]**2))
    return K


def smo_simple_kernel(data, labels, C, toler, max_iter, K):
    """
        smo算法简化核化版
        :param data: 数据集
        :param labels: 标签集
        :param C: 常数C
        :param toler: 容错率
        :param max_iter: 最大循环次数
        :param K: 核化数据
        :return: b和alphas
        """
    # 数据预处理
    data = np.array(data)
    labels = np.array(labels)
    m, n = data.shape

    # 初始化b和alphas
    b = 0
    alphas = np.zeros(m)

    iter = 0  # 存储没有alpha改变的情况下遍历数据集的次数
    while iter < max_iter:
        a_changed = 0  # 记录alpha是否进行优化
        # 遍历样本
        for i in range(m):
            yi_ = float(np.dot(alphas * labels, K[i])) + b  # xi的预测值：(w.T)Xi+b
            ei = yi_ - float(labels[i])  # 计算与真实值误差，若较大则要对αi进行更改
            # 判断是否更改α
            if ((labels[i] * ei < -toler) and alphas[i] < C) or ((labels[i] * ei > toler) and alphas[i] > 0):
                j = select_j(i, m)  # 选取另一个α
                yj_ = float(np.dot(alphas * labels, K[i])) + b  # xj的预测值
                ej = yj_ - float(labels[j])

                ai_old = alphas[i].copy()  # 记录原来的αi的值
                aj_old = alphas[j].copy()  # 记录原来的αj的值

                # 计算L和H，保证alpha在0与C之间
                if labels[i] != labels[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] - alphas[i])

                # L=H则不改变
                if L == H:
                    print('L==H')
                    continue

                # 计算αj的最优修改量
                eta = 2.0 * K[i, j] - K[j, j] - K[i, i]
                if eta >= 0:
                    print("eta>=0")
                    continue

                # 对αj进行调整
                alphas[j] -= labels[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - aj_old) < 0.00001:
                    print('j not move enough')
                    continue
                # 对αi做同αj相反方向变化
                alphas[i] += labels[i] * labels[j] * (aj_old - alphas[j])

                # 设置常数项b
                b1 = b - ei - labels[i] * (alphas[i] - ai_old) * K[i, i] - labels[j] * (
                            alphas[j] - aj_old) * K[j, j]
                b2 = b - ej - labels[i] * (alphas[i] - ai_old) * K[i, j] - labels[j] * (
                            alphas[j] - aj_old) * K[j, j]
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                a_changed += 1
                print("iter:{} i:{}, pairs changed: {}".format(iter, i, a_changed))

        if a_changed == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number:{}'.format(iter))

        return b, alphas


def test_kernel(labels, alphas, b, K):
    err_cnt = 0
    for i in range(len(labels)):
        predict = K[i].T @ np.multiply(alphas, labels) + b
        if labels[i] * predict < 0:
            err_cnt += 1
    return err_cnt / len(labels)
