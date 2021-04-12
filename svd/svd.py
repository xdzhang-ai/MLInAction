"""
svd的应用 2020年05月27日11:11:38
"""
import numpy as np


def load_ex_data():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 1, 0, 0]]


"""
相似度计算
################
"""


def euclid(veca, vecb):
    """
    基于欧式距离
    :param veca:
    :param vecb:
    :return:
    """
    return 1.0/(1.0 + np.linalg.norm(veca - vecb))


def pearson(veca, vecb):
    """
    基于皮尔逊相关系数
    :param veca:
    :param vecb:
    :return:
    """
    if len(veca) < 3:
        return 1.0
    return 0.5 + 0.5*np.corrcoef(veca, vecb, rowvar=False)[0][1]


def cosine(veca, vecb):
    """
    基于余弦相似度
    :param veca:
    :param vecb:
    :return:
    """
    return 0.5 + 0.5*(veca @ vecb / (np.linalg.norm(veca) * np.linalg.norm(vecb)))

"""
################
"""


def stand_est(data, user, sim_means, item):
    """
    计算在给定相似度计算方法条件下，用户对物品的估计评分值
    :param data:
    :param user:
    :param sim_means:
    :param item: 未评分物品
    :return:
    """
    n = data.shape[1]
    # 初始化总相似度和评分加权相似度
    sim_total = 0.0
    rat_sim_total = 0.0
    for j in range(n):
        user_rating = data[user, j]
        if user_rating == 0:
            continue
        overlap = np.nonzero(np.logical_and(data[:, item] > 0, data[:, j] > 0))[0]
        if len(overlap) == 0:
            similarity = 0
        else:
            # 对评分过的物品计算相似度
            similarity = sim_means(data[overlap, item], data[overlap, j])
        # 更新相似度
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def svd_est(data, user, sim_means, item):
    """
    将data经svd分解降维后估计评分
    :param data:
    :param user:
    :param sim_means:
    :param item:
    :return:
    """
    n = data.shape[1]
    # 初始化总相似度和评分加权相似度
    sim_total = 0.0
    rat_sim_total = 0.0

    U, sigma, VT = np.linalg.svd(data)
    sig4 = np.eye(4) * sigma[: 4]
    # 重构数据
    data_rec = data.T @ U[:, :4] @ np.linalg.inv(sig4)
    for j in range(n):
        user_rating = data[user, j]
        if user_rating == 0 or j == item:
            continue
        similarity = sim_means(data_rec[item, :], data_rec[j, :])
        # 更新相似度
        sim_total += similarity
        rat_sim_total += similarity * user_rating
    if sim_total == 0:
        return 0
    else:
        return rat_sim_total / sim_total


def recommend(data, user, N=3, sim_meas=cosine, est_method=stand_est):
    """
    产生N个推荐结果
    :param data:
    :param user:
    :param N:
    :param sim_meas: 相似度计算方法
    :param est_method: 估计评分方法
    :return:
    """
    # 对给定用户建立未评分物体列表
    unrated_items = np.squeeze(np.argwhere(data[user, :] == 0))
    if len(unrated_items) == 0:
        return 'you rated everything'
    item_scores = []
    for item in unrated_items:
        estimated_score = est_method(data, user, sim_meas, item)
        item_scores.append((item, estimated_score))
    return sorted(item_scores, key=lambda jj: jj[1], reverse=True)[:N]



