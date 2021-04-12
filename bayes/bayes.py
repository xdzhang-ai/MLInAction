"""
贝叶斯分类 2020-04-22 15:30:19
"""
import numpy as np


def load_data():
    """
    生成词向量数据
    :return: 词向量和对应的标签，1表示侮辱性，0表示正常
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec


def create_vocab(data):
    """
    创建一个data中包含的不重复的词汇表
    :param data: 词向量数据
    :return: 不重复词汇表
    """
    vocab = set([])     # 创建空集合
    for document in data:
        vocab = vocab | set(document)   # 求并集

    return list(vocab)


def doc2vec_set(vocab, doc):
    """
    生成一个能反映给定文档中的词汇在词汇表中出现情况的向量(词集模型：单词是否出现)
    :param vocab: 词汇表
    :param doc: 文档
    :return: 0，1向量
    """
    vec = [0]*len(vocab)    # 初始化向量，将所有值置0，表示词汇表中的所有词汇都未在文档中出现,长度为词汇表的长度

    # 遍历文档中的所有单词
    for word in doc:
        if word in vocab:
            index = vocab.index(word)
            vec[index] = 1          # 若单词在词汇表中出现，则向量相应位置置1
    return vec


def doc2vec_bag(vocab, doc):
    """
    生成一个能反映给定文档中的词汇在词汇表中出现情况的向量(词袋模型：单词出现次数)
    :param vocab: 词汇表
    :param doc: 文档
    :return: 数值向量
    """
    vec = [0]*len(vocab)    # 初始化向量，将所有值置0，表示词汇表中的所有词汇都未在文档中出现,长度为词汇表的长度

    # 遍历文档中的所有单词
    for word in doc:
        if word in vocab:
            index = vocab.index(word)
            vec[index] += 1          # 若单词在词汇表中出现，则向量相应位置置1
    return vec


def trainNB0(train_X, train_Y):
    """
    根据贝叶斯定理 p(ci|w)=p(w|ci)p(ci)/p(w)，该函数实现条件概率p(w|ci)在假定单词wj独立情况下的计算
    :param train_X: 文档矩阵，由doc2vec返回的01向量组成的m×n矩阵，m为文档数，n为单词个数
    :param train_Y: 文档矩阵标签，m×1。1表示侮辱性，0表示正常
    :return: 二分类的条件概率向量p(wj|c0), p(wj|c1)及先验概率p(c1)
    """
    # 计算先验概率P(C1)
    m = len(train_Y)        # 文档（样本）个数
    n = len(train_X[0])     # 词汇表单词个数
    pc1 = sum(train_Y) / m

    # 计算条件概率
    # 初始化各类中不同单词统计值向量(分子)
    c0_w_cnt = np.ones(n)
    c1_w_cnt = np.ones(n)
    # 初始化各类中出现的单词总数(分母)
    c0_sum = 2
    c1_sum = 2
    # 遍历文档
    for i in range(m):
        if train_Y[i] == 1:         # 统计标签为侮辱的文档中每个单词出现的个数
            c1_w_cnt += train_X[i]
            c1_sum += sum(train_X[i])
        else:
            c0_w_cnt += train_X[i]     # 统计标签为正常的文档中每个单词出现的个数
            c0_sum += sum(train_X[i])
    # 计算概率
    pw_c1 = c1_w_cnt / c1_sum       # 除以标签为侮辱的文档中单词总数
    pw_c0 = c0_w_cnt / c0_sum  # 除以标签为正常的文档中单词总数

    return np.log(pw_c0), np.log(pw_c1), pc1        # 将结果取对数防止多个极小数相乘而导致的下溢出


def classifyNB(vec, pw_c0, pw_c1, pc1):
    """
    对待分类向量分类
    :param vec: 待分类向量
    :param pw_c0: 训练所得的c0条件概率
    :param pw_c1: 训练所得的c1条件概率
    :param pc1: 训练所得的c1先验概率
    :return: 类别
    """
    # 根据贝叶斯定理计算后验概率
    p1 = np.sum(vec * pw_c1) + np.log(pc1)  # 取对数后连乘变累加，累加元素只含vec中含有的单词（查表）
    p0 = np.sum(vec * pw_c0) + np.log(1-pc1)

    if p1 > p0:
        return 1
    return 0


def test_NB():
    """
    整合上述流程，封装所有操作
    :return:
    """
    # 获取数据
    data, train_Y = load_data()
    # 根据数据生成词汇表
    vocab = create_vocab(data)
    # 创建训练文档矩阵
    train_X = []
    for doc in data:
        train_X.append(doc2vec_set(vocab, doc))
    # 训练数据
    pw_c0, pw_c1, pc1 = trainNB0(train_X, train_Y)

    # 分类
    test_x1 = ['love', 'my', 'dalmation']
    test_vec1 = doc2vec_set(vocab, test_x1)
    print("result:{}".format(classifyNB(test_vec1, pw_c0, pw_c1, pc1)))

    test_x2 = ['tupid', 'garbage']
    test_vec2 = doc2vec_set(vocab, test_x2)
    print("result:{}".format(classifyNB(test_vec2, pw_c0, pw_c1, pc1)))