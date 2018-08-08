import re

from bs4 import BeautifulSoup
import nltk
import numpy as np
import pandas as pd
from gensim.models import word2vec, Word2Vec
from sklearn.ensemble import RandomForestClassifier

SENTENCE_LIMIT_SIZE = 120
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3


def loadDataSet(filePath):
    """
    将tsv文件读取出来
    :param filePath:
    :return:
    """
    return pd.read_csv(filePath, header=0, delimiter='\t', quoting=3)


def getWordList(review):
    """
    将review转换成向量的形式
    :param review:
    :return:
    """
    with open("./data/english") as f:
        text = f.read()
        stopwords = text.splitlines()

    # 去除HTML标签
    review_text = BeautifulSoup(review).get_text()
    # 去除非法字符
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    # 转换成小写并分割成列表
    words = review_text.lower().split()
    # 去除停用词
    words = [word for word in words if word not in stopwords]
    return words


def reviewToSentences(review):
    """
    将评论分段处理
    :param review:
    :param tokenizer:
    :return:
    """
    raw_sentences = review.strip().split(',')
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(getWordList(review))

    return sentences


def saveModel():
    train = loadDataSet('./data/labeledTrainData.tsv')
    unlabeledTrain = loadDataSet('./data/unlabeledTrainData.tsv')
    sentences = []
    for review in train['review']:
        sentences += reviewToSentences(review)

    for review in unlabeledTrain['review']:
        sentences += reviewToSentences(review)
        
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count=min_word_count,
                              window=context, sample=downsampling)
    model.init_sims(replace=True)
    model_name = "300features_40minwords_10context"
    model.save(model_name)


def makeFeatureVec(words, model, num_features):
    """

    :param words:
    :param model:
    :param num_features: 词向量的大小
    :return:
    """
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2wordSet = set(model.index2word)
    for word in words:
        if word in index2wordSet:
            nwords += 1
            featureVec = np.add(featureVec, model[word])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    """

    :param reviews:
    :param model:
    :param num_features:
    :return:
    """
    counter = 0
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    for review in reviews:
        reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)

    return reviewFeatureVecs


def main():
    train = loadDataSet('./data/labeledTrainData.tsv')
    test = loadDataSet('./data/testData.tsv')
    model = Word2Vec.load("300features_40minwords_10context")

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


saveModel()
main()