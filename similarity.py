import numpy as np
import math

def standard_deviation(sim):
    """计算标准差
    :param sim:
    :return:区分度值
    """
    length = len(sim) - 1
    temp = 0
    for i in sim:
        temp += i
    ave = (temp - 1) / length
    sum = 0
    for i in sim:
        sum += pow((i - ave), 2)
    result = (sum - pow((1 - ave), 2)) * 1 / length
    return math.sqrt(result)


def similarity(lda_sim, tfidf_sim, lam):
    """计算加权相似度
    :return:相似度矩阵
    """
    return lam * lda_sim + (1 - lam) * tfidf_sim


def lamdba_func(standard1, standard2):
    return standard1 / (standard1 + standard2)


def cos_sim(doc_dis1, doc_dis2):
    """文本相似度计算
    :param doc_dis1: 文档1
    :param doc_dis2: 文档2
    :return: 相似度列表
    """
    user_dis = np.mat(doc_dis1)
    doc_dis = np.mat(doc_dis2)
    num = float(user_dis * doc_dis.T)
    denom = np.linalg.norm(user_dis) * np.linalg.norm(doc_dis)
    cos = num / denom
    result = cos
    return result


if __name__ == "__main__":
    # lda_sim = np.load('lda_sim_topic_20.npy')
    # tfidf_sim = np.load('tfidf_sim.npy')
    # standard1 = 0
    # standard2 = 0
    #
    # for i in lda_sim:
    #     standard1 += standard_deviation(i)
    # for j in tfidf_sim:
    #     standard2 += standard_deviation(j)
    # standard1 = standard1 / 1599
    # standard2 = standard2 / 1599
    # print(standard1, standard2)
    # param_lamdba = lamdba_func(standard1, standard2)
    # print(param_lamdba)
    # sim = similarity(lda_sim, tfidf_sim, param_lamdba)
    # np.save('sim.npy', sim)
    # a = np.load('fre.npy')
    # #
    tf_idf = np.load('chinese_l_tfidf.npy')

    raw1 = []
    for i in range(0, len(tf_idf)):
        print('正在计算第 %i 篇与其他文章的相似度。。。' % i)
        raw3 = []
        for j in tf_idf:
            result = cos_sim(tf_idf[i], j)
            raw3.append(result)
        raw1.append(raw3)
    np.save('chinese_gamma_0.4_tfidf_sim.npy', raw1)

