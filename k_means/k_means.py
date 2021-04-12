"""
K均值聚类算法的实现及应用 2020-05-16 15:26:42
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


def cal_distance(vec_a, vec_b):
    """
    计算两个向量的欧式距离
    :param vec_a:
    :param vec_b:
    :return:
    """
    return np.sqrt(np.sum((vec_a - vec_b)**2))


def rand_center(data, k):
    """
    初始化随机质心
    :param data:
    :param k: 簇数
    :return:
    """
    n = data.shape[1]
    centroids = np.zeros((k, n))

    for i in range(n):
        min_i = min(data[:, i])
        range_i = max(data[:, i]) - min_i
        centroids[:, i] = min_i + range_i * np.random.rand(k)

    return centroids


def k_means(data, k, dist=cal_distance, init_center=rand_center):
    """
    k均值
    :param data:
    :param k:
    :param dist: 距离计算方式
    :param init_center: 初始化质心方式
    :return:
    """
    m = data.shape[0]
    cluster_assign = np.zeros((m, 2))   # 存储每个点的簇分配结构，第一列记录簇索引，第二列记录点到簇质心的距离
    centroids = init_center(data, k)    # 初始化质心

    cluster_changed = True  # 当簇不在变化时停止迭代
    while cluster_changed:
        cluster_changed = False
        # 对每一个点寻找最近质心
        for i in range(m):
            min_dis = np.inf
            min_idx = -1
            for j in range(k):
                dist_ij = dist(data[i], centroids[j])
                if dist_ij < min_dis:
                    min_dis = dist_ij
                    min_idx = j
            # 有一个点的分簇结果变化就继续迭代
            if cluster_assign[i][0] != min_idx:
                cluster_changed = True
            cluster_assign[i] = min_idx, min_dis**2
        print(centroids)
        # 更新质心
        for cent in range(k):
            cent_idxs = cluster_assign[:, 0] == cent
            centroids[cent, :] = np.mean(data[cent_idxs], axis=0)

    return centroids, cluster_assign


def bi_K_means(data, K, dist=cal_distance):
    """
    二分K均值聚类
    :param data:
    :param K:
    :param dist:
    :return:
    """
    cent0 = np.mean(data, axis=0)
    m = data.shape[0]
    clusters = np.zeros((m, 2))
    cent_list = [cent0]
    for i in range(m):
        clusters[i][1] = dist(data[i], cent0)

    while len(cent_list) < K:
        lowest_sse = np.inf
        for i in set(clusters[:, 0]):
            centroids, clusters_new = k_means(data[clusters[:, 0] == i], 2)    # 对当前簇一分为二

            sse_new = np.sum(clusters_new[:, 1])    # 计算当前簇二分后的sse
            sse_other = np.sum(clusters[clusters[:, 0] != i][:, 1])

            if sse_new + sse_other < lowest_sse:
                best_idx = i
                best_clus = clusters_new
                lowest_sse = sse_new + sse_other
                best_centroids = centroids


        best_clus[best_clus[:, 0] == 0, 0] = len(cent_list)
        best_clus[best_clus[:, 0] == 1, 0] = best_idx
        clusters[clusters[:, 0] == best_idx] = best_clus

        cent_list[int(best_idx)] = best_centroids[0]
        cent_list.append(best_centroids[1])

    return cent_list, clusters