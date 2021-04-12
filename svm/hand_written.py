"""
svm识别手写数字 2020-05-02 15:01:57
"""
import numpy as np
import os
from svm import create_kernel, smo_simple_kernel


def img2vec(file):
    """二进制图像矩阵转向量"""
    with open(file) as f:
        lines = f.readlines()
    s = ''
    for line in lines:
        s += line[:32]

    # 将32*32的图片张成向量
    vec = np.array(list(s), dtype=int).reshape(1, 1024)
    return vec


def load_data(file, k_tup):
    """加载数据"""
    # 文件路径
    path = os.getcwd() + '/' + file
    files = os.listdir(path)

    # 读取数据集
    labels = []
    matrix = img2vec(path+'/'+files[0])

    label = int(files[0].split('_')[0])
    labels.append(label)
    for img in files[1:]:
        matrix = np.concatenate((matrix, img2vec(path+'/'+img)))
        label = int(img.split('_')[0])
        if label == 9:
            labels.append(-1)
        else:
            labels.append(1)

    m = len(labels)
    K = np.zeros((m, m))
    for i in range(m):
        K[i] = create_kernel(matrix, matrix[i], k_tup)

    return matrix, np.array(labels), K


def test_kernel(svs, data, labels, label_sv, alpha_sv, b, k_tup):
    err_cnt = 0
    m, n = data.shape
    for i in range(m):
        kernel = create_kernel(svs, data[i], k_tup)
        predict = kernel.T @ np.multiply(label_sv, alpha_sv) + b
        if labels[i] * predict < 0:
            err_cnt += 1
    return err_cnt / m


def test_digits(k_tup=('rbf', 10)):
    data, labels, K = load_data('trainingDigits', k_tup)
    b, alphas = smo_simple_kernel(data, labels, 200, 0.0001, 10000, K)

    sv_idx = np.nonzero(alphas > 0)[0]
    svs = data[sv_idx]
    label_sv = labels[sv_idx]
    alpha_sv = alphas[sv_idx]

    print('-----------')
    print('训练集错误率：')
    print(test_kernel(svs, data, labels, label_sv, alpha_sv, b, k_tup))

    print('-------------')
    print('测试集错误率：')
    data, labels = load_data('testDigits', k_tup)[:2]
    print(test_kernel(svs, data, labels, label_sv, alpha_sv, b, k_tup))
