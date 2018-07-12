import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA


def deleteFeat(df):
    """
    删除一些不相关或者有很多缺失值的特征
    1, 删除缺失值较多，切无关紧要的列， 如‘MiscFeature’，‘LotFrontage’, Alley, Fence, FireplaceQu。
    2, 删除一些特征相关明显的特征，如GarageYrBlt， TotRmsAbcGrd，1stFlrSF，GarageArea，OverallCond
    3, 删除一些和输出结果不相关的特征，如MiscVal，YrSold，BsmtFinSF2， MasVnrArea，MasVnrType, LowQualFinSF
    4, 删除一些具有明显偏重的值，如PoolArea， PoolQC, MiscVal
    :param df:
    :return:
    """
    columns = ['MiscFeature', 'LotFrontage', 'Alley', 'Fence', 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars',
               'OverallCond', 'MasVnrArea', 'BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', 'MiscVal', 'YrSold',
               'PoolArea', 'PoolQC', 'FireplaceQu', 'MasVnrType']
    dfDelete = df.drop(columns, axis=1)

    return dfDelete


def fillMissingData(df):
    """
    填补缺失值的重要特征
    :param df:
    :return:
    """
    df.loc[df.GarageCond.isnull(), ['GarageCond', 'GarageType',
                                    'GarageFinish', 'GarageQual']] = 'noGarage'

    df.loc[df.BsmtExposure.isnull(), ['BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']] = 'noBasement'

    return df


def normalDistribution(dataSet, type):
    dataSet['GrLivArea'] = np.log1p(dataSet['GrLivArea'])
    dataSet['TotalBsmtSF'] = np.log1p(dataSet['TotalBsmtSF'])

    if type == 'train':
        dataSet['SalePrice'] = np.log1p(dataSet['SalePrice'])

    return dataSet


def dummiesFeat(df):
    df = pd.get_dummies(df, dummy_na=True)

    # 对月份做离散化处理
    bins1 = pd.IntervalIndex.from_tuples([(0, 3), (3, 6), (6, 9), (9, 12)])
    df['MoSold'] = pd.cut(df['MoSold'], bins1)
    dummiesMoSold = pd.get_dummies(df['MoSold'], prefix='MoSold')

    # 对整体的质量和条件做离散化处理
    # bins2 = pd.IntervalIndex.from_tuples([(0, 3), (3, 6), (6, 10)])
    # df['OverallCond'] = pd.cut(df['OverallCond'], bins2)
    # df['OverallQual'] = pd.cut(df['OverallQual'], bins2)
    # dummiesOverallCond = pd.get_dummies(df['OverallCond'], prefix='OverallCond')
    # dummiesOverallQual = pd.get_dummies(df['OverallQual'], prefix='OverallCond')

    train = pd.concat([df, dummiesMoSold], axis=1)

    train.drop(['MoSold'], axis=1, inplace=True)

    return train


def tackle_dataSet():
    data_train = pd.read_csv('./data/train.csv')
    data_test = pd.read_csv('./data/test.csv')

    len_train = len(data_train)

    dataSet = pd.concat([data_train, data_test], axis=0)

    dataSet = deleteFeat(dataSet)
    dataSet = dummiesFeat(dataSet)

    train_data = dataSet[:len_train]
    test_data = dataSet[len_train:]

    train_data.drop([948, 1379], axis=0, inplace=True)
    train_data = normalDistribution(train_data, 'train')
    test_data = normalDistribution(test_data, 'test')
    trainY = train_data['SalePrice']

    trainX = train_data.drop(['Id', 'SalePrice'], axis=1)

    testX = test_data.fillna(0)
    testX = testX.drop(['Id', 'SalePrice'], axis=1)

    train_len = len(trainX)

    data = pd.concat([trainX, testX], axis=0)

    data = pca(data)

    train = data[:train_len]
    test = data[train_len:]
    # train = pca(trainX)
    # test = pca(testX)

    return train, trainY, test, test_data['Id']


def pca(dataSet):
    pca = PCA(n_components=15)
    pca.fit(dataSet)
    dataSet = pca.transform(dataSet)

    return dataSet


def norm(dataSet):
    scaler = MinMaxScaler()
    scaler.fit(dataSet)
    trainXNorm = scaler.transform(dataSet)

    return trainXNorm

from sklearn.ensemble import GradientBoostingRegressor
def train(trainX, trainY):

    trainXNorm = norm(trainX)

    trainData, validateData, trainLabels, validateLabels = train_test_split(trainXNorm, trainY, test_size=0.3, random_state=0)

    # reg = RandomForestRegressor(n_estimators=800, max_depth=15, min_samples_split=3, max_features='sqrt', random_state=0)
    reg = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                    max_depth=4, max_features='sqrt',
                                    min_samples_leaf=15, min_samples_split=10,
                                    loss='huber', random_state=5)

    reg.fit(trainData, trainLabels)

    validations = reg.score(validateData, validateLabels)

    return reg, validations


def test(testX, reg, Id):

    testXNorm = norm(testX)
    predictions = reg.predict(testXNorm)

    result = pd.DataFrame({'Id': Id.as_matrix(), 'SalePrice': np.exp(predictions.astype(np.float32))})

    result.to_csv('./data/rfg_prediction.csv', index=False)


if __name__ == "__main__":
    import time
    start = time.time()
    data_train, train_labels, data_test, test_Id = tackle_dataSet()
    reg, scores = train(data_train, train_labels)
    test(data_test, reg, test_Id)
    print(scores)
    print(time.time() - start)
