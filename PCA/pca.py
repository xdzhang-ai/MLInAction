"""
PCA算法实现及应用 2020-05-26 11:23:22
"""
import numpy as np


def load_data(file, split='\t'):
    with open(file) as f:
        lines = f.readlines()

    num_feat = len(lines[0].strip().split(split))
    data = []
    for line in lines:
        line = line.strip().split(split)
        vec = []
        for i in range(num_feat):
            vec.append(float(line[i]))

        data.append(vec)

    return np.array(data)


def pca(data, feat_num):
    """
    pca
    :param data:
    :param feat_num: 需要降成几维
    :return:
    """
    # 去平均
    mean_data = np.mean(data, axis=0)
    data = data - mean_data

    # 计算协方差矩阵
    cov_mat = np.cov(data, rowvar=False)
    # 计算特征值与特征向量
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # 对特征值排序
    sort_idx = np.argsort(eig_vals)
    eig_idx = sort_idx[-feat_num:]
    # 选择相应的特征向量
    eig_vecN = eig_vecs[:, eig_idx]

    # 计算降维后的数据
    data_new = data @ eig_vecN
    # 重构数据
    recov_data = data_new @ eig_vecN.T + mean_data

    return data_new, recov_data