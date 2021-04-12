"""
利用朴素贝叶斯实现垃圾邮件分类 2020-04-23 12:51:59
"""
from bayes import create_vocab, doc2vec_set, trainNB0, classifyNB
import random


def parse_text(string):
    """
    文本解析，对邮件内容进行解析，生成词向量
    :param string: 邮件内容
    :return: 词向量
    """

    import re
    word_vec = re.split(r'\W+', string)         # 分隔符为除单词和数字的任意字符串
    return [word.lower() for word in word_vec if len(word) > 2]         # 保留长度大于2的小写单词


def test_email():
    """
    测试错误率
    :return:  测试集错误率
    """
    data = []   # 数据集，每个元素都是一个词向量
    labels = []     # 标签集，存放数据标签

    for i in range(1, 26):
        with open('email/ham/{}.txt'.format(i), encoding='ISO-8859-1') as f:
            email = f.read()
            data.append(parse_text(email))
            labels.append(1)
        with open('email/spam/{}.txt'.format(i), encoding='ISO-8859-1') as f:
            email = f.read()
            data.append(parse_text(email))
            labels.append(0)

    vocab = create_vocab(data)      # 根据数据集创建词汇表

    train_set = [i for i in range(50)]   # 训练集和测试集的所有索引值
    test_set = []
    # 选择10个索引作为测试集索引
    for i in range(20):
        test_idx = int(random.uniform(0, len(train_set)))
        test_set.append(test_idx)
        del(train_set[test_idx])

    # 剩下的作为训练集
    train_X = []
    train_Y = []
    for idx in train_set:
        train_X.append(doc2vec_set(vocab, data[idx]))
        train_Y.append(labels[idx])
    # 训练
    pw_c0, pw_c1, pc1 = trainNB0(train_X, train_Y)

    # 测试
    error_cnt = 0
    for idx in test_set:
        vocab_vec = doc2vec_set(vocab, data[idx])
        res = classifyNB(vocab_vec, pw_c0, pw_c1, pc1)
        if res != labels[idx]:
            error_cnt += 1

    print("错误率：{}".format(error_cnt/len(test_set)))


if __name__ == '__main__':
    test_email()