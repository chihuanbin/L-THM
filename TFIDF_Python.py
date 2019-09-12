from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import os
import chardet
import jieba
from nltk import word_tokenize

class Tfidf_Python(object):
    """
        LDA和TF-IDF改进计算文本相似度方法
    """

    def __init__(self):
        self.seg_list = []
        self.tf_idf_value = None
        self.tf_idf_corpus = None
        self.corpus = []
        self.gamma = []
        self.vocab = None
        self.tfidf_ppCountMatrix = None
        self.l_weigth = None
        self.key_word = []

    def process_chinese_data(self):
        """
        加载中文预料，并进行分词，去停用词操作，数据的预处理
        :return:None
        """
        # 读入每篇文档并进行处理
        print("开始加载文档...")
        merage_filder = os.getcwd() + '/corpus'  # 获取当前路径
        file_name = os.listdir(merage_filder) # 文本名称列表
        for name in file_name:
            with open('Corpus' + '/' + name, 'rb') as f:
                text = f.read()
            text_decode = text.decode(encoding='gb2312', errors='ignore')
            seg_generator = jieba.lcut(text_decode, cut_all=True) #进行中文分词
            self.corpus.append(' '.join(seg_generator))
            self.seg_list.append([i for i in seg_generator])
            print("加载文档 %s 成功！" % name)
        np.save('chinese_tfidf_corpus.npy', self.corpus)
        print('全部文档加载完成！！')
        print('-------------------------')

    def process_english_data(self):
        """
        加载英文预料，并进行分词，去停用词操作，数据的预处理
        :return: None
        """
        print('开始加载英文文档...')
        merage_filder = os.getcwd() + '/20newsgroup'  # 获取当前路径
        file_name = os.listdir(merage_filder)
        print(file_name)
        for name in file_name:
            with open('20newsgroup' + '/' + name, 'rb') as f:
                raw = f.read()
                result = chardet.detect(raw)
            with open('20newsgroup' + '/' + name, 'r', encoding=result['encoding']) as f:
                text = f.read()
            self.corpus.append(text)
            self.seg_list.append(word_tokenize(text))
            print("加载文档 %s " % name)
        np.save('english_tfidf_corpus.npy', self.corpus)
        print('加载文档完成')
        print('---------------------------------')

    def tf_idf(self):
        """
        计算TFIDF
        :return: None
        """
        with open('stopwords.txt', 'r', encoding='utf-8') as f:
            line = f.readlines()

        stopwords = []
        print('开始计算tfidf值')
        # vectorizer = CountVectorizer(stop_words='english', max_df=0.7) #  英文预料所需要的参数
        vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=stopwords)  # 中文预料所需要的参数 #将文本中的词语转换成词频矩阵
        transformer = TfidfTransformer()#计算每个词的tf-idf权值
        count = vectorizer.fit_transform(self.corpus) #文本转化为词频矩阵
        self.tf_idf_value = transformer.fit_transform(count)#计算tf-idf
        self.weigth = self.tf_idf_value.toarray()  # 转为tf idf矩阵
        self.word = vectorizer.get_feature_names()  # 所有文本的关键词
        np.save('tfidf.npy', self.weigth)
        print('计算TFIDF完成！！！\n')
        print('---------------------------------')

    def save_wiegth_english_word_file(self, key):
        """
        保存权重词
        :return: None
        """
        index_sort = []  # 排序后返回的每篇文章元素的索引
        length = []  # 每篇文章的长度
        key_word = [] # 取出的前gamma个关键词
        words_index = [] # 取出的每篇文章的排序后所有关键词的索引
        for doc in self.seg_list:
            length.append(len(doc))
        f = open('weigth_gamma_word.txt', 'w', encoding='utf-8')
        print('计算gamma值并找出关键词')
        print('--------------------')
        print('对每篇文章的词权重进行排序')
        for i in range(0, len(self.seg_list)):
            indexs = np.argsort(-self.tf_idf_value.getrow(i).data, axis=0)
            index_sort.append(indexs)
        print('排序完成')
        print('---------------------')
        print('查找前gamma个权重词的索引')
        for i, j in zip(range(0, len(index_sort) + 1), index_sort):
            # print(tf.getrow(i).indices[j])
            words_index.append(self.tf_idf_value.getrow(i).indices[j])
        print('查找完成')
        print('---------------------')
        print('计算每篇文章的gamma值')
        for j in length:
            raw = int(j * key)
            self.gamma.append(raw)
        print('gamma值计算完成')
        print('---------------------')
        print('取出每篇文章前gamma个关键词')
        key_word = []
        for doc, length in zip(words_index, self.gamma):
            for i in doc[:length]:
                key_word.append(self.word[i])
                f.write(self.word[i])
                f.write('\n')
        f.close()
        self.vocab = list(set(key_word))
        # for index in word_index:
        #     for i in index:
        #         key_word.append(self.word[i])
        #         f.write(self.word[i])
        #         f.write('\n')
        # f.close()
        # self.vocab = list(set(key_word))
        np.save('weigth_gamma_word_stopwors.npy', self.vocab)
        print('取出并保存完成！！！')
        print('---------------------')

    def save_wiegth_chinese_word_file(self, key):
        """
        保存权重词
        :return: None
        """
        index_sort = []  # 排序后返回的每篇文章元素的索引
        length = []  # 每篇文章的长度
        words_index = [] # 取出的每篇文章的排序后所有关键词的索引
        for doc in self.seg_list:
            length.append(len(doc))
        f = open('weigth_gamma_word.txt', 'w', encoding='utf-8')
        print('计算gamma值并找出关键词')
        print('--------------------')
        print('对每篇文章的词权重进行排序')
        for i in range(0, len(self.seg_list)):
            indexs = np.argsort(-self.tf_idf_value.getrow(i).data, axis=0)
            index_sort.append(indexs)
        print('排序完成')
        print('---------------------')
        print('查找前gamma个权重词的索引')
        for i, j in zip(range(0, len(index_sort) + 1), index_sort):
            # print(tf.getrow(i).indices[j])
            words_index.append(self.tf_idf_value.getrow(i).indices[j])
        print('查找完成')
        print('---------------------')
        print('计算每篇文章的gamma值')
        for j in length:
            raw = int(j * key)
            self.gamma.append(raw)
        print('gamma值计算完成')
        print('---------------------')
        print('取出每篇文章前gamma个关键词')
        for doc, length in zip(words_index, self.gamma):
            for i in doc[:length]:
                self.key_word.append(self.word[i])
                f.write(self.word[i])
                f.write(' ')
            f.write('\n')
        f.close()
        np.save('key_word.npy', self.key_word)
        print('取出并保存完成！！！')
        print('---------------------')


    def english_count_freq(self):
        """
        计算文本在提取的权重词下的频数，用于计算tfidf权重值
        :return:
        """
        print('正在统计每篇文章提取关键词的词频...')
        counter = []
        for document in self.seg_list:
            count = np.zeros(len(self.vocab), dtype=np.int)
            for word in document:
                if word in self.vocab:
                    count[self.vocab.index(word)] = count[self.vocab.index(word)] + 1
            counter.append(count)
        self.tfidf_ppCountMatrix = np.array(counter)
        np.save('tfidf_ppCountMatrix.npy', self.tfidf_ppCountMatrix)
        print('保存改进tf完成')
        print('---------------------------------')

    def englsih_l_tfidf(self):
        """
        通过提取出的关键词计算TFIDF
        :return: None
        """
        print('计算改进tfidf值')
        l_transformer = TfidfTransformer()
        l_tfidf = l_transformer.fit_transform(self.tfidf_ppCountMatrix)
        self.l_weigth = l_tfidf.toarray()  # 改进的tf idf矩阵
        np.save('english_l_tfidf.npy', self.l_weigth)
        print('计算改进TFIDF完成！！！\n')

    def chinese_l_tfidf(self):

        print('计算改进tfidf值')
        with open('weigth_gamma_word.txt', 'r', encoding='utf-8') as f:
            corpus = f.readlines()
            l_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.8)#将词语转换为词频矩阵
            l_transformer = TfidfTransformer() #计算每个词的tf-idf
            count = l_vectorizer.fit_transform(corpus) #将文本转换为词频矩阵
            l_tfidf = l_transformer.fit_transform(count) #计算tf-idf 值
            self.l_weigth = l_tfidf.toarray()  # 改进的tf idf矩阵
            np.save('chinese_l_tfidf.npy', self.l_weigth)
            print('计算改进TFIDF完成！！！\n')


def main():
    tfidf_process = Tfidf_Python()
    tfidf_process.process_chinese_data()
    tfidf_process.tf_idf()
    tfidf_process.save_wiegth_chinese_word_file(0.4)
    tfidf_process.chinese_l_tfidf()

if __name__ == '__main__':
    main()

