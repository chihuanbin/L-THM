import lda
import jieba
import re
import numpy as np
import os
import chardet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import codecs

class Lda_Python(object):
    """

    """
    def __init__(self):
        """
        构造函数
        """
        self.stop_words = []  # 停用词表
        self.seg_list = []  # 词矩阵，一行代表一个文档的所有词，用于统计词频
        self.seglist = []  # 词列表，整个分词处理的预料
        self.word_bag = []  # LDA词袋
        self.ppCountMatrix = None  # 词频矩阵
        self.model = None  # lda模型

    def load_stop_word(self, filename):
        """
        加载停用词
        :param filename:停用词文件名
        :return: None
        """
        print('加载停用词...')
        with open(file=filename, mode='br') as f:
            stop_word = f.read()
            self.stop_words = stop_word.decode(encoding='utf-8', errors='ignore')
        print("加载停用词表 %s 完成!" % filename)

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
            seg_generator = jieba.cut(text_decode, cut_all=False)
            seg_generator = [w for w in seg_generator if not re.match('^[a-z|A-Z|0-9|._ \x21-\x7e]*$', w)]
            self.seglist.extend([i for i in seg_generator if i not in self.stop_words])
            self.seg_list.append([i for i in seg_generator if i not in self.stop_words])
            print("加载文档 %s 成功！" % name)
        # np.save('seg_list.npy', self.seg_list)
        # np.save('seglist.npy', self.seglist)
        print('全部文档加载完成！！')

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
            seg_generator = word_tokenize(text)
            seg_generator = [w for w in seg_generator if w.lower() not in self.stop_words]
            seg_generator = [w for w in seg_generator if not re.match('^[0-9]', w) and len(w) >= 3]
            self.seglist.extend([i for i in seg_generator if i.lower() not in stopwords.words('english')])
            self.seg_list.append([i for i in seg_generator if i.lower() not in stopwords.words('english')])
            print("加载文档 %s " % name)
        # np.save('english_seg_list.npy', self.seg_list)
        # np.save('english_seglist.npy', self.seglist)
        print('加载文档完成')

    def create_word_bag(self):
        """
        去除重复的词,保存LDA词袋
        :return: 词列表
        """
        print('去除重复的词并保存词列表...')
        for word in self.seglist:
            if (word not in self.word_bag and word != None):
                self.word_bag.append(word)
        np.save('lda_word_bag.npy', self.word_bag)
        print('保存词袋成功')

    def count_matrix(self):
        """
        统计词频，用于计算lda
        :return: None
        """
        print('正在统计所有关键词词频...')
        counter = []
        for document in self.seg_list:
            count = np.zeros(len(self.word_bag), dtype=np.int)
            for word in document:
                if word in self.word_bag:
                    count[self.word_bag.index(word)] += 1
            counter.append(count)
        self.ppCountMatrix = np.array(counter)  # 得到词频矩阵
        np.save('ppCountMatrix.npy', self.ppCountMatrix)
        print('统计词频并保存完成')

    def fit_model(self, n_topic, n_iter, alpha, eta):
        """
        训练LDA模型
        :param n_topic: 主题个数
        :param n_iter: 迭代次数
        :param alpha: LDA参数α
        :param eta: LDA参数η
        :return: None
        """
        self.model = lda.LDA(n_topics=n_topic, n_iter=n_iter, alpha=alpha, eta=eta, random_state=1)
        self.model.fit(self.ppCountMatrix)

    def print_topic_word(self, n_top_word=8):
        """
        打印主题和词分布矩阵
        :param n_top_word: 要打印的主题下词的个数
        :return: None
        """
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.word_bag)[np.argsort(topic_dist)][:-(n_top_word + 1):-1]
            print("Topic:", i, "\t"),
            for word in topic_words:
                print(word)

    def save_topic_words(self, n_top_word=-1):
        """
        保存主题词矩阵
        :param n_top_word:
        :return:
        """
        if n_top_word == -1:
            n_top_word = len(self.word_bag)
        f = codecs.open('topic_words.txt', 'w', 'utf-8')
        for i, topic_dist in enumerate(self.model.topic_word_):
            topic_words = np.array(self.word_bag)[np.argsort(topic_dist)][:-(n_top_word + 1):-1]
            f.write("Topic:%d\t" % i)
            for word in topic_words:
                f.write("%s " % word)
            f.write("\n")
        f.close()

    def save_doc_topic(self):
        """
        保存文档主题模型
        :return: None
        """
        topic_document = []
        f = codecs.open('doc_topic.txt', 'w', 'utf-8')
        for i in range(len(self.ppCountMatrix)):
            #  f.write("Doc %d:((top topic:%s) topic distribution:%s)\n" %
            #  (i, self.model.doc_topic_[i].argmax(), self.model.doc_topic_[i]))
            f.write(str(self.model.doc_topic_[i]) + '\n')
            topic_document.append(self.model.doc_topic_[i])
        f.close()
        np.save('doc_topic.npy', topic_document)

def main():
    lda_process = Lda_Python()
    lda_process.load_stop_word('stopwords.txt')
    lda_process.process_chinese_data()
    lda_process.create_word_bag()
    lda_process.count_matrix()
    lda_process.fit_model(80, 100, 0.1, 0.15)

if __name__ == '__main__':
    main()