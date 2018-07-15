import re

from bs4 import BeautifulSoup
import nltk
import numpy as np
import pandas as pd


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



