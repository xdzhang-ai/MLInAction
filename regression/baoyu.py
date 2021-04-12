"""
鲍鱼年龄预测 2020-05-09 15:30:03
"""
import numpy as np
from regression import lwlr_test, load_data


def square_error(y, y_s):
    """
    计算均方误差
    :param y:
    :param y_s:
    :return:
    """
    return sum((y - y_s)**2)


def ridge_regres(data, labels, lam=0.2):
    """
    岭回归
    :param data:
    :param labels:
    :param lam: 岭回归系数λ
    :return:
    """
    n = data.shape[1]
    xtx = data.T @ data
    xtx = xtx + lam * np.eye(n)

    if np.linalg.det(xtx) == 0:
        print("矩阵不可逆")
        return -1

    w = np.linalg.inv(xtx) @ data.T @ labels
    return w


def ridge_test(x, y):
    """
    岭回归测试
    :param x:
    :param y:
    :return:
    """
    # 数据标准化
    y_mean = y.mean()
    y = y - y_mean

    x_mean = x.mean(0)
    x_var = x.var(0)
    x = (x - x_mean) / x_var

    num_lam = 30
    w_mat = np.zeros((num_lam, x.shape[1]))

    for i in range(num_lam):
        w_mat[i] = ridge_regres(x, y, np.exp(i-10))
    return w_mat


def stage_wise(x, y, eps=0.01, num_it=100):
    """
    前向逐步线性回归
    :param x:
    :param y:
    :param eps:
    :param num_it:
    :return:
    """
    # 数据标准化
    y_mean = y.mean()
    y = y - y_mean
    x_mean = x.mean(0)
    x_var = x.var(0)
    x = (x - x_mean) / x_var

    m, n = x.shape
    ws = np.zeros(n)
    w_list = np.zeros((num_it, n))

    for i in range(num_it):
        # 初始化最小误差为正无穷
        min_error = np.inf
        w_best = np.zeros(n)
        # 对每个特征
        for j in range(n):
            # 增大或减小
            for step in [-1, 1]:
                ws_test = ws.copy()
                ws_test[j] += step * eps
                # 计算新的系数下预测值
                y_ = x @ ws_test
                # 计算新的误差值
                error = square_error(y, y_)

                if error < min_error:
                    w_best = ws_test
                    min_error = error

        ws = w_best
        w_list[i] = ws

    return w_list
