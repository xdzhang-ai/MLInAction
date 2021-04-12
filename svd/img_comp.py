"""
svd应用于图像压缩 2020-05-27 17:03:36
"""
import numpy as np


def print_img(img, thresh=0.8):
    """
    根据阈值打印图像
    :param img:
    :param thresh:
    :return:
    """
    for i in range(32):
        for k in range(32):
            if float(img[i,k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print()


def img_compress(num_sv=3, thresh=0.8):
    """
    图像压缩
    :param num_sv:
    :param thresh:
    :return:
    """
    img = []
    with open('0_5.txt') as f:
        lines = f.readlines()

    for line in lines:
        row = []
        for i in range(32):
            row.append(int(line[i]))
        img.append(row)
    img = np.array(img)

    print('初始图像')
    print_img(img, thresh)

    # svd
    U, sigma, VT = np.linalg.svd(img)

    sigma_sv = sigma[: num_sv] * np.eye(num_sv)
    # 重构图像
    reco_img = U[:, :num_sv] @ sigma_sv @ VT[:num_sv, :]
    print('重构图像')
    print_img(reco_img, thresh)