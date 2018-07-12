import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
                              ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from mlxtend.classifier import StackingClassifier


def loadData():
    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')

    return data_train, data_test


def firstDrop(dataSet):
    """
    去掉首次分析后不需要的特征
    :param dataSet:
    :return:
    """
    dataSet['hasCabin'] = dataSet['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
    dataSet = dataSet.drop(['Cabin', 'Ticket'], axis=1)

    return dataSet


def fromNameExtractInfo(dataSet):
    """
    从Name特征中提取Title属性的值
    :param dataSet:
    :return:
    """
    dataSet['Title'] = dataSet.Name.str.extract(r'([A-Za-z]+)\.', expand=False)
    dataSet['Title'] = dataSet['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
                                                 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataSet['Title'] = dataSet['Title'].replace('Mlle', 'Miss')
    dataSet['Title'] = dataSet['Title'].replace('Ms', 'Miss')
    dataSet['Title'] = dataSet['Title'].replace('Mme', 'Mrs')

    title_mapping = {'Mr': 1, 'Miss': 4, 'Mrs': 5, 'Master': 3, 'Rare': 2}

    dataSet['Title'] = dataSet['Title'].map(title_mapping)
    dataSet['Title'] = dataSet['Title'].fillna(0)

    dataSet = dataSet.drop(['Name'], axis=1)

    return dataSet


def sexTransform(dataSet):
    """
    将性别转换成数字表示
    :param dataSet:
    :return:
    """
    dataSet['Sex'] = dataSet['Sex'].map({'female': 1, 'male': 0}).astype(int)

    return dataSet


def ageTransform(dataSet):
    """
    先将年龄中的缺失值补全，然后将年龄分段转变成离散值
    :param dataSet:
    :return:
    """
    guess_ages = np.zeros((2, 3))

    for i in range(2):
        for j in range(3):
            guess_df = dataSet[(dataSet['Sex'] == i) & (dataSet['Pclass'] == j + 1)]['Age'].dropna()
            guess_ages[i, j] = int(guess_df.median() / 0.5 + 0.5) * 0.5

    for i in range(2):
        for j in range(3):
            dataSet.loc[(dataSet['Sex'] == i) & (dataSet['Pclass'] == j + 1) & (dataSet.Age.isnull()), 'Age'] = \
            guess_ages[i, j]
    dataSet['Age'] = dataSet['Age'].astype(int)

    dataSet.loc[dataSet['Age'] <= 16, 'Age'] = 0
    dataSet.loc[(dataSet['Age'] > 16) & (dataSet['Age'] <= 32), 'Age'] = 1
    dataSet.loc[(dataSet['Age'] > 32) & (dataSet['Age'] <= 48), 'Age'] = 2
    dataSet.loc[(dataSet['Age'] > 48) & (dataSet['Age'] <= 64), 'Age'] = 3
    dataSet.loc[dataSet['Age'] > 64, 'Age'] = 4

    return dataSet


def extractSibSpAndParch(dataSet):
    """
    将SibSp和Parch合并在一起
    :param dataSet:
    :return:
    """
    dataSet['FamilySize'] = dataSet['SibSp'] + dataSet['Parch'] + 1
    dataSet['IsAlone'] = 0
    dataSet.loc[dataSet['FamilySize'] == 1, 'IsAlone'] = 1
    dataSet = dataSet.drop(['Parch', 'SibSp'], axis=1)

    return dataSet


def megerPclassAndAge(dataSet):
    """
    因为年龄和等级有一定的关系，可以将两者合并在一起使用
    :param dataSet:
    :return:
    """
    dataSet['AgePclass'] = dataSet.Age * dataSet.Pclass

    return dataSet


def embarkedTransform(dataSet):
    """
    先补全Embarked的缺失值，然后对其进行数值化转换
    :param dataSet:
    :return:
    """
    freq_port = dataSet.Embarked.dropna().mode()[0]
    dataSet.loc[dataSet.Embarked.isnull(), 'Embarked'] = freq_port

    dataSet['Embarked'] = dataSet['Embarked'].map({'S': 0, 'Q': 1, 'C': 2}).astype(int)

    return dataSet


def fareTransform(dataSet):
    """
    将Fare值划分成离散值
    :param dataSet:
    :return:
    """
    dataSet.loc[dataSet['Fare'] <= 7.91, 'Fare'] = 0
    dataSet.loc[(dataSet['Fare'] > 7.91) & (dataSet['Fare'] <= 14.454), 'Fare'] = 1
    dataSet.loc[(dataSet['Fare'] > 14.454) & (dataSet['Fare'] <= 31), 'Fare'] = 2
    dataSet.loc[dataSet['Fare'] > 31, 'Fare'] = 3
    dataSet['Fare'] = dataSet['Fare'].astype(int)

    return dataSet


def dummy(dataSet):
    dummyAge = pd.get_dummies(dataSet['Age'], prefix='Age')
    # dummyPclass = pd.get_dummies(dataSet['Pclass'], prefix='Pclass')
    dummySex = pd.get_dummies(dataSet['Sex'], prefix='Sex')
    dummyEmbarked = pd.get_dummies(dataSet['Embarked'], prefix='Embarked')
    dummyTitle = pd.get_dummies(dataSet['Title'], prefix='Title')
    dummyIsAlone = pd.get_dummies(dataSet['IsAlone'], prefix='IsAlone')
    dummyFare = pd.get_dummies(dataSet['Fare'], prefix='Fare')

    dataSet = pd.concat([dataSet, dummyEmbarked, dummyFare, dummyIsAlone, dummyAge, dummySex, dummyTitle],
                        axis=1)
    dataSet = dataSet.drop(['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'Fare'], axis=1)

    return dataSet


def brain(dataSet):
    """
    将上述对数据集处理的方法集成到一起
    :param dataSet:
    :return:
    """
    dataSet = firstDrop(dataSet)

    dataSet = fromNameExtractInfo(dataSet)

    dataSet = sexTransform(dataSet)

    dataSet = ageTransform(dataSet)

    dataSet = extractSibSpAndParch(dataSet)

    dataSet = megerPclassAndAge(dataSet)

    dataSet = embarkedTransform(dataSet)

    dataSet = fareTransform(dataSet)

    # dataSet = dummy(dataSet)

    return dataSet


def train(data_train, clf):
    """
    训练模型
    :param data_train:
    :param clf:
    :return:
    """
    trainX = data_train.drop(['PassengerId', 'Survived'], axis=1)
    trainY = data_train['Survived']

    trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.1, random_state=0)

    clf.fit(trainX.values, trainY.values)
    scores = clf.score(validX.values, validY.values)

    return scores, clf


def test(data_test, clf):
    """
    测试模型
    :param data_test:
    :param clf:
    :return:
    """
    testX = data_test.drop(['PassengerId'], axis=1)
    prections = clf.predict(testX.values)
    result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': prections})
    result.to_csv('./data/prections2.csv', index=False)


def main():

    data_train, data_test = loadData()
    data_train = brain(data_train)
    data_test = brain(data_test)
    data_train = data_train.drop(['AgePclass'], axis=1)
    data_test = data_test.drop(['AgePclass'], axis=1)
    models = {
        'LogisticReg': LogisticRegression(max_iter=500, tol=0.0001, penalty='l2', solver='lbfgs'),
        'svc': SVC(max_iter=200, kernel='rbf', gamma=0.5, C=5),
        'KNN': KNeighborsClassifier(n_neighbors=9),
        'LinearSvc': LinearSVC(max_iter=250, penalty='l2', C=0.5),
        'decisionTree': DecisionTreeClassifier(max_depth=4),
        'randomTree': RandomForestClassifier(n_estimators=1000, n_jobs=-1, min_samples_leaf=2,
                                             random_state=0),
        'gbdt': GradientBoostingClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, random_state=0),
        'adaboost': AdaBoostClassifier(n_estimators=300, learning_rate=0.75, random_state=0),
        'extract': ExtraTreesClassifier(n_estimators=250, n_jobs=-1, max_depth=5, random_state=0),
        'gnb': GaussianNB(),
    }

    # stackModel = StackingClassifier(classifiers=[models['decisionTree'],
    #                                                 models['gbdt'], models['adaboost'],
    #                                                 models['extract']],
    #                                    meta_classifier=models['randomTree'])

    # for key in models:
    #     scores, clf = train(data_train, models[key])
    #
    #     print("model: {0}   scores: {1}".format(key, scores))
    # clf = SVC(max_iter=200, kernel='rbf', gamma=0.5, C=5)
    scores, clf = train(data_train, models['KNN'])
    test(data_test, clf)
    print(scores)


class StackingAverageModels(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    自定义stacking模型类，用于堆砌其他的模型成为一个新的模型，目的在于集中每个模型的有点
    """
    def __init__(self, base_models: tuple, meta_model, n_folds=5):
        self.base_models = base_models  # 用于对训练集和测试集输出新的特征的模型
        self.meta_model = meta_model  # 用于对提取出的特征做预测过
        self.n_folds = n_folds

    def fit(self, X, y):
        """
        在此类中总共有两层的训练
        :param X:
        :param y:
        :return:
        """
        self.base_models_ = [[] for x in self.base_models]  # 用来存储所有训练过的模型，并将同类模型存入到一个列表中
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=0)

        # 创建一个数组用于存储每个模型的对训练集进行交叉预测的结果
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        # 依次取出每一个模型对数据进行预测
        for i, model in enumerate(self.base_models):
            # 将训练集进行进行分割
            for train_index, holdout_index in kfold.split(X, y):
                # 因为每个模型都会进行5次训练，而且每次训练的训练集都不一样，因此每次都对未fit的模型克隆一份来进行操作
                instance = clone(model)
                instance.fit(X[train_index], y[train_index])
                self.base_models_[i].append(instance)
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred

        # 将之前测试的结果作为新的特征用作训练集，训练集的输出值不变
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        """
        对测试集进行测试，先用slef.base_models_中保存的模型对测试集进行预测，
        将同一类模型的预测值取平均值生成一个新的特征，然后用所有不同模型生成的特征作为新的输入值
        :param X:
        :return:
        """
        meta_feature = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_
        ])

        return self.meta_model_.predict(meta_feature)


if __name__ == '__main__':
    main()


