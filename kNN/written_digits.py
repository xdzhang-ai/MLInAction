import numpy as np
import os
from kNN import classify0


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


def load_data(file):
    """加载数据"""
    # 文件路径
    path = os.getcwd() + '/digits/' + file
    files = os.listdir(path)

    # 读取数据集
    labels = []
    matrix = img2vec(path+'/'+files[0])

    label = int(files[0].split('_')[0])
    labels.append(label)
    for img in files[1:]:
        matrix = np.concatenate((matrix, img2vec(path+'/'+img)))
        label = int(img.split('_')[0])
        labels.append(label)

    return matrix, np.array(labels)


def class_test():
    """测试数据"""
    train, train_labels = load_data('trainingDigits')
    test, test_labels = load_data('testDigits')
    m = test.shape[0]

    error_cnt = 0
    for i in range(m):
        predict = classify0(test[i], train, train_labels, 3)
        if predict != test_labels[i]:
            error_cnt += 1
        print("predict:{}   real:{}\n".format(predict, test_labels[i]))

    print("测试集总数为{}，错误个数为{}，错误率为{}%\n".format(m, error_cnt, error_cnt/float(m)*100))


if __name__ == '__main__':
    class_test()
