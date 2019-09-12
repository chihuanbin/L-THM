# -*- enconding:utf-8 -*-

import time
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
start = time.time()

weight = np.load('chinese_gamma_0.4_tfidf_sim.npy')
weight= preprocessing.scale(weight)    #数据标准化
# weight = preprocessing.normalize(weight, norm='l2')
data = pd.DataFrame(weight)    #转换为表
kmeans = KMeans(n_clusters=8)   #构造聚类器
kmeans.fit(data)   #训练聚类模型

class_number = 300
all_counter = Counter(kmeans.labels_)
labels_pred = kmeans.labels_.tolist()   #转为列表

a = kmeans.labels_   #给出样本标签
print(a[:300])

true_labels = []
for i in range(0, 8):
    for j in range(0, 300):
        true_labels.append(i)


f1_all = []
for i in range(0,2400,300):
    train_counter = Counter(labels_pred[i:i + 300])   #获取labels_pred[i]到[i+300]
    print(Counter(labels_pred[i:i + 300]).most_common(8))
    class_num = train_counter.most_common(1)   #第一个标签和数目，也是最多的数目
    top_doc = class_num[0][0]   #取出最多的数目的标签
    precision = class_num[0][1] / all_counter[top_doc]
    recall = class_num[0][1] / class_number
    f1 = (2 * precision * recall) / (precision + recall)
    f1_all.append(f1)

print(all_counter)
print(f1_all)

g =1 / 8
count = 0
for i in f1_all:
    count += g * i
print(count)
end = time.time()

print('运行时间 %d' % (end - start))

