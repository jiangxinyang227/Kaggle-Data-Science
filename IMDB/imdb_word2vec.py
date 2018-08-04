import random
import re
import time
from collections import Counter

from bs4 import BeautifulSoup
import numpy as np
import tensorflow as tf


class ImdbWord2Vec:
    def __init__(self, train_review, unlabeled_review):

        self.vocab_size = None
        self.train_review = None
        self.unlabeled_review = None
        self.freq = 5
        self.vocab_to_int = None

    def word_to_token(self):
        """
        生成词映射表，并且去除词频过低的词，去除特殊字符
        :param train_review:
        :param unlabeled_review:
        :return: 处理后的全样本组成的词向量
        """
        train_review = self.train_review
        unlabeled_review = self.unlabeled_review
        freq = self.freq

        text = " ".join(train_review + unlabeled_review)

        # 去除HTML标签
        text = BeautifulSoup(text).get_text()
        # 去除非法字符
        text = text.replace('.', ' <PERIOD> ')
        text = text.replace(',', ' <COMMA> ')
        text = text.replace('"', ' <QUOTATION_MARK> ')
        text = text.replace(';', ' <SEMICOLON> ')
        text = text.replace('!', ' <EXCLAMATION_MARK> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        text = text.replace('(', ' <LEFT_PAREN> ')
        text = text.replace(')', ' <RIGHT_PAREN> ')
        text = text.replace('--', ' <HYPHENS> ')
        text = text.replace('?', ' <QUESTION_MARK> ')
        # text = text.replace('\n', ' <NEW_LINE> ')
        text = text.replace(':', ' <COLON> ')
        # 转换成小写并分割成列表
        words = text.lower().split()

        # 删除低频次，减少噪音
        word_counts = Counter(words)
        trimmed_words = [word for word in words if word_counts[word] > freq]

        return trimmed_words

    def create_new_data(self, words, t=1e-5, thres=0.8):
        """
        基于处理后的词向量构建词向量映射表，将原始数据集转换成由数值表示的，并且进行采样处理，去除停用词，去除低频词汇
        :param words:
        :return: 用于训练的词和词映射表的大小
        """
        vocab = set(words)
        vocab = list(vocab)
        self.vocab_size = len(vocab)
        vocab_to_int = {w: c for w, c in enumerate(vocab)}
        self.vocab_to_int = vocab_to_int
        total_count = len(words)
        word_freqs = {w: c/total_count for w, c in vocab_to_int}
        prob_drop = {w: 1-np.sqrt(t / word_freqs[w]) for w in vocab_to_int}
        words = [w for w in words if prob_drop[w] < thres]

        int_words = [vocab_to_int[w] for w in words]

        return int_words

    def get_targets(self, words, index, window_size=5):
        """
        获取输出值，首先确定中心值的index，然后确定上下文的位置，然后构成输出值
        :param words:
        :param index:
        :param window_size:
        :return:
        """
        target_window = np.random.randint(1, window_size+1)
        start_point = index - target_window if (index - target_window) > 0 else 0
        end_point = index + target_window

        targets = set(words[start_point: index] + words[index+1: end_point+1])
        return list(targets)

    def get_batches(self, words, batch_size, window_size=5):
        """
        获得mini-batch样本输出
        :param words:
        :param batch_size:
        :param window_sizes:
        :return:
        """
        n_batches = len(words) // batch_size

        # 仅取full batches
        words = words[: n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx: idx + batch_size]
            for i in range(len(batch)):
                batch_x = batch[i]
                batch_y = self.get_targets(batch, i, window_size)
                # 由于一个input word会对应多个output word，因此需要长度统一
                x.extend([batch_x] * len(batch_y))
                y.extend(batch_y)
            yield x, y

    def get_embedding_word(self, n_sampled=100, embedding_size=200):
        """
        训练获取词向量
        :param n_sampled:
        :param embedding_size:
        :return:
        """
        train_graph = tf.Graph()
        with train_graph.as_default():
            inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')

            # 嵌入层权重矩阵
            embedding = tf.Variable(tf.random_uniform([self.vocab_size, embedding_size], -1, 1))
            # 实现lookup
            embed = tf.nn.embedding_lookup(embedding, inputs)

            softmax_w = tf.Variable(tf.truncated_normal([self.vocab_size, embedding_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.vocab_size))

            # 计算negative sampling下的损失
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, self.vocab_size)

            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
            normalized_embedding = embedding / norm

        epochs = 10  # 迭代轮数
        batch_size = 1000  # batch大小
        window_size = 10  # 窗口大小

        saver = tf.train.Saver()  # 文件存储

        words = self.word_to_token()
        train_words = self.create_new_data(words)

        with tf.Session(graph=train_graph) as sess:
            iteration = 1
            loss = 0
            sess.run(tf.global_variables_initializer())

            for e in range(1, epochs + 1):
                batches = self.get_batches(train_words, batch_size, window_size)
                start = time.time()
                #
                for x, y in batches:

                    feed = {inputs: x,
                            labels: np.array(y)[:, None]}
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                    loss += train_loss

                    if iteration % 100 == 0:
                        end = time.time()
                        print("Epoch {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Avg. Training loss: {:.4f}".format(loss / 100),
                              "{:.4f} sec/batch".format((end - start) / 100))
                        loss = 0
                        start = time.time()

            save_path = saver.save(sess, "checkpoints/text8.ckpt")
            embed_mat = sess.run(normalized_embedding)

        return embed_mat
