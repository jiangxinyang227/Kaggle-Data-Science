import warnings

import numpy as np
import pandas as pd

from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
from sklearn.model_selection import GridSearchCV


warnings.filterwarnings('ignore')


def loadData():
    """
    产生数据集
    :return:
    """
    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')

    return data_train, data_test


def deleteOutlier(train):
    """
    删除训练集中的异常点
    :param train:
    :return:
    """
    train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)

    return train


def imputingMissingData(dataSet):
    """
    对整个数据集进行缺失值的补全
    :param dataSet: 测试集和训练集的组合
    :return:
    """
    dataSet['PoolQC'] = dataSet['PoolQC'].fillna('None')
    dataSet['MiscFeature'] = dataSet['MiscFeature'].fillna('None')
    dataSet['Alley'] = dataSet['Alley'].fillna('None')
    dataSet['Fence'] = dataSet['Fence'].fillna('None')
    dataSet['FireplaceQu'] = dataSet['FireplaceQu'].fillna('None')

    # 根据Neighborhood特征将LotFrontage进行分类，然后针对每个类别进行去中位数去补全缺失值
    dataSet['LotFrontage'] = dataSet.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

    for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
        dataSet[col] = dataSet[col].fillna('None')

    for col in ('GarageArea', 'GarageCars'):
        dataSet[col] = dataSet[col].fillna(0)

    dataSet['GarageYrBlt'] = dataSet['GarageYrBlt'].fillna(method='ffill')

    for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
        dataSet[col] = dataSet[col].fillna('None')

    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
        dataSet[col] = dataSet[col].fillna(0)

    dataSet['MasVnrType'] = dataSet['MasVnrType'].fillna('None')
    dataSet['MasVnrArea'] = dataSet['MasVnrArea'].fillna(0)

    dataSet.drop(['Utilities'], axis=1, inplace=True)  # 该特征在测试集中的取值一样

    # 缺失值很少的值，而且也不是像前面的特征中那样是因为不存在而为Nan，类似这种的都可以用众数来补全
    dataSet['MSZoning'] = dataSet['MSZoning'].fillna(dataSet['MSZoning'].mode()[0])

    dataSet['Functional'] = dataSet['Functional'].fillna(dataSet['Functional'].mode()[0])

    dataSet['Electrical'] = dataSet['Electrical'].fillna(dataSet['Electrical'].mode()[0])

    dataSet['KitchenQual'] = dataSet['KitchenQual'].fillna(dataSet['KitchenQual'].mode()[0])

    dataSet['Exterior1st'] = dataSet['Exterior1st'].fillna(dataSet['Exterior1st'].mode()[0])
    dataSet['Exterior2nd'] = dataSet['Exterior2nd'].fillna(dataSet['Exterior2nd'].mode()[0])

    dataSet['SaleType'] = dataSet['SaleType'].fillna(dataSet['SaleType'].mode()[0])

    return dataSet


def transformStr(dataSet):
    """
    将一些类别型的特征中的数字值转换成字符串
    :param dataSet:
    :return:
    """
    dataSet['MSSubClass'] = dataSet['MSSubClass'].apply(str)
    dataSet['OverallCond'] = dataSet['OverallCond'].apply(str)
    dataSet['YrSold'] = dataSet['YrSold'].apply(str)
    dataSet['MoSold'] = dataSet['MoSold'].apply(str)

    return dataSet


def transformSortNum(dataSet):
    """
    将类别型特征中的值转换成序列型数字, 例如["jiang", "zhang", "jiang"] -> [0, 1, 0]
    :param dataSet:
    :return:
    """
    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
            'ExterQual', 'ExterCond', 'HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
            'YrSold', 'MoSold')
    for col in cols:
        lbl = LabelEncoder()
        lbl.fit(list(dataSet[col].values))
        dataSet[col] = lbl.transform(list(dataSet[col].values))

    return dataSet


def extractFeat(dataSet):
    """
    从其他特征中提取新特征，将地下室和第一层，第二层的面积加在一起，构成一个新的特征
    :param dataSet:
    :return:
    """
    dataSet['TotalSF'] = dataSet['TotalBsmtSF'] + dataSet['1stFlrSF'] + dataSet['2ndFlrSF']

    return dataSet


def boxCoxFeat(dataSet):
    """
    skew大于0.75的将数值型特征都进行box-cox变换成服从正态分布的特征
    :param dataSet:
    :return:
    """
    numericalFeat = dataSet.dtypes[dataSet.dtypes != 'object'].index

    skewed_feats = dataSet[numericalFeat].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    skewness = pd.DataFrame({'Skew': skewed_feats})

    skewness = skewness[abs(skewness) > 0.75]  # 将斜度的绝对值大于0.75的特征进行BoxCox转换

    skewedFeatures = skewness.index
    lam = 0.15
    for feature in skewedFeatures:
        dataSet[feature] = boxcox1p(dataSet[feature], lam)

    return dataSet


def dummy(dataSet):
    """
    将一些类别型的特征转换成稀疏的数值型特征
    :param dataSet:
    :return:
    """
    dataSet = pd.get_dummies(dataSet)

    return dataSet


def train(trainX, trainY, model):
    """
    训练模型
    :param trainX:
    :param trainY:
    :param model:
    :return:
    """
    # trainData, validateData, trainLabels, validateLabels = train_test_split(trainX, trainY, test_size=0.2, random_state=0)
    # model.fit(trainData, trainLabels)
    #
    # scores = model.score(validateData, validateLabels)
    model.fit(trainX, trainY)
    scores = model.score(trainX, trainY)

    return scores, model


def test(testX, testId, model):
    """
    测试模型
    :param testX:
    :param testId:
    :param model:
    :return:
    """
    predictions = model.predict(testX)

    result = pd.DataFrame({'Id': testId.as_matrix(), 'SalePrice': np.exp(predictions).astype(np.float32)})

    result.to_csv('./data/rfg_prediction.csv', index=False)


def main():

    data_train, data_test = loadData()

    data_train = deleteOutlier(data_train)
    trainX = data_train.drop(['Id', 'SalePrice'], axis=1)
    trainY = data_train['SalePrice']
    trainY = np.log1p(trainY)
    ntrain = len(trainX)

    testId = data_test['Id']
    testX = data_test.drop(['Id'], axis=1)

    dataSet = pd.concat([trainX, testX], axis=0)

    dataSet = imputingMissingData(dataSet)
    dataSet = transformStr(dataSet)
    dataSet = transformSortNum(dataSet)
    dataSet = extractFeat(dataSet)
    dataSet = boxCoxFeat(dataSet)
    dataSet = dummy(dataSet)

    trainX = dataSet[:ntrain]
    testX = dataSet[ntrain:]

    models = {
        # "lgd": lgb.LGBMRegressor(objective='regression', num_leaves=5,
        #                          learning_rate=0.05, n_estimators=720,
        #                          max_bin=55, bagging_fraction=0.8,
        #                          bagging_freq=5, feature_fraction=0.2319,
        #                          feature_fraction_seed=9, bagging_seed=9,
        #                          min_data_in_leaf=6, min_sum_hessian_in_leaf=11),

        'gbdt': GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=12,
                                          loss='huber', random_state=0),
        "ridge": Ridge(alpha=0.05, max_iter=100),

        'krr': KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5),

        'enet': ElasticNet(alpha=0.0005, l1_ratio=1, random_state=3),

        'lasso': Lasso(alpha=0.0008, max_iter=100, random_state=1),

    }
    stackReg = StackingRegressor(regressors=[models['gbdt'], models['krr'], models['enet']], meta_regressor=models['lasso'])
    modelStack1 = StackingAverageModels(base_models=(models['ridge'], models['krr'], models['enet']),
                                        meta_model=models['lasso'])

    modelStack2 = StackingAverageModels(base_models=(models['gbdt'], models['ridge'], models['krr'], models['lasso']),
                                        meta_model=models['enet'])

    modelStack3 = StackingAverageModels(base_models=(models['gbdt'], models['lasso'], models['enet']),
                                        meta_model=models['krr'])

    modelStack4 = StackingAverageModels(base_models=(models['lasso'], models['krr'], models['enet']),
                                        meta_model=models['gbdt'])

    averageModel = AveragingModels((modelStack1, modelStack2, modelStack3, modelStack4))
    # for key in models:
    #     scores, reg = train(trainX, trainY, models[key])
    #
    #     print('model: {}   scores: {}'.format(key, scores))

    # parameters = {"n_estimators": [500], "min_samples_split": [12, 15], "min_samples_leaf": [12, 15],
    #                 "max_depth": [4, 6], "random_state": [0]}
    # clf = GridSearchCV(models['gbdt'], parameters)

    scores, reg = train(trainX.values, trainY.values, modelStack2)
    test(testX.values, testId, reg)
    print(scores)


# stacking models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    简单的对多个模型进行融合，对每个模型的预测结果去平均值，从而达到更好的预测结果
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        # clone方法能对模型对象进行深拷贝， 因此下面实际上是对我们的模型集合进行了深度拷贝
        self.models_ = [clone(x) for x in self.models]

        for model in self.models_:
            """
            对每个模型进行训练，训练模型没有返回值
            """
            model.fit(X, y)
        return self

    def predict(self, X):
        """
        用fit函数训练号的模型来调用该方法，对测试集进行预测
        :param X:
        :return:
        """
        # 获得的predictions应该是每行为一个模型的预测数据
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        # 将所有模型预测的结果取一个平均值，对于回归的处理方式
        return np.mean(predictions, axis=1)


class StackingAverageModels(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    自定义stacking模型类，用于堆砌其他的模型成为一个新的模型，目的在于集中每个模型的有点
    """
    def __init__(self, base_models: tuple, meta_model, n_folds=10):
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


main()