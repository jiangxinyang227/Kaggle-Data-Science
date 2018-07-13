import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
import os


def download_data(dir):
    data = {}
    data['sentence'] = []
    for file in os.listdir(dir):
        with open(os.path.join(dir, file), 'r') as f:
            data['sentence'].append(f.read())
    return pd.DataFrame.from_dict(data)


def load_data(dir):
    pos_df = download_data(os.path.join(dir, 'pos'))
    neg_df = download_data(os.path.join(dir, 'neg'))
    pos_df['sentiment'] = 1
    neg_df['sentiment'] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_indec(drop=True)


train_df = load_data('/home/jiangxinyang/git_projects/machine_learning/tensorflow/Sequences/aclImdb/train')
test_df = load_data('/home/jiangxinyang/git_projects/machine_learning/tensorflow/Sequences/aclImdb/test')


