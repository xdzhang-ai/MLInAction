"""
马是否存活 2020-05-06 15:56:19
"""
from adaboost import adaboost_train, ada_classify, plot_roc
import numpy as np


def load_data(file):
    data = []
    labels = []

    with open(file) as f:
        num_feat = len(f.readline().split('\t'))
    with open(file) as f:
        lines = f.readlines()

    for line in lines:
        content = line.strip().split('\t')
        lin_arr = []
        for i in range(num_feat-1):
            lin_arr.append(float(content[i]))
        data.append(lin_arr)
        labels.append(float(content[-1]))

    return np.array(data), labels