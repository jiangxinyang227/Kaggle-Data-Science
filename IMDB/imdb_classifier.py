import re

from bs4 import BeautifulSoup
import nltk
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from .imdb_word2vec import ImdbWord2Vec

SENTENCE_LIMIT_SIZE = 120
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3


def loadDataSet(filePath):
    """
    将tsv文件读取出来
    :param filePath:
    :return: 返回df数据集
    """
    return pd.read_csv(filePath, header=0, delimiter='\t', quoting=3)


def convert_text_to_token(sentence, word_to_token_map, limit_size=SENTENCE_LIMIT_SIZE):
    """
    根据单词-编码映射表将单个句子转化为token
    return: 句子转换为token后的列表
    """
    # 获取unknown单词和pad的token
    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]

    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split()]

    # Pad
    if len(tokens) < limit_size:
        # 选择填充0，其实就是对应了"<pad>"
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]

    return tokens


def getWordList(reviews, word_to_token_map):
    """
    将数据集转换成固定长度的序列，然后转换成词向量
    :param reviews:
    :return:
    """
    newReview = []
    for review in reviews:
        newReview.append(convert_text_to_token(review, word_to_token_map))

    return np.array(newReview)


def fetchNewData():
    train = loadDataSet('./data/labeledTrainData.tsv')
    test = loadDataSet('./data/testData.tsv')
    train_review = train.review.tolist()

    train_x = getWordList(train_review, word_to_token_map)
    test_x = getWordList(test.review.tolist(), word_to_token_map)
    train_y = np.array(train.sentimet.tolist())


def main():
    unlabeled_train = loadDataSet('./data/unlabeledTrainData.tsv')
    train = loadDataSet('./data/labeledTrainData.tsv')
    test = loadDataSet('./data/testData.tsv')

    unlabeled_review = unlabeled_train.review.tolist()
    train_review = train.review.tolist()

    model = ImdbWord2Vec(train_review, unlabeled_review)
    word2vec = model.get_embedding_word()
    word_to_token_map = model.vocab_to_int

    train_x = getWordList(train_review, word_to_token_map)
    test_x = getWordList(test.review.tolist(), word_to_token_map)
    train_y = np.array(train.sentimet.tolist())

    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(getWordList(review))

    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model, num_features)

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(getWordList(review))

    testDataVecs = getAvgFeatureVecs(clean_test_reviews, model, num_features)

    classifier = RandomForestClassifier(n_estimators=200, max_features="sqrt", max_depth=8, min_samples_split=4)
    classifier.fit(trainDataVecs, train['sentiment'])
    result = classifier.predict(testDataVecs)
    output = pd.DataFrame(data={"id": test['id'], "sentiment": result})
    output.to_csv("./data/Word2Vec_AverageVectors.csv", index=False, quoting=3)