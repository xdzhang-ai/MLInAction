"""
PCA处理半导体数据 2020-05-26 13:30:11
"""
import numpy as np
from pca import load_data, pca


def nan_process():
    """
    缺失值处理，设置为平均值
    :return:
    """
    data = load_data('secom.data', ' ')
    num_feat = data.shape[1]
    for i in range(num_feat):
        column = data[:, i]
        nan_idx = np.isnan(column)
        mean_val = np.mean(column[~nan_idx])
        data[:, i][nan_idx] = mean_val

    return data


