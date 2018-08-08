import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import xgboost as xgb


def load():
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    # 去除attributed_time字段,该字段和is_attributed重合
    x_train = pd.read_csv("./data/train.csv", nrows=10000000, parse_dates=['click_time'], dtype=dtypes,
                          usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    return x_train


def predispose(x_train):
    x_train['day'] = x_train['click_time'].dt.day.astype('uint8')
    x_train['hour'] = x_train['click_time'].dt.hour.astype('uint8')

    gp = x_train[['ip', 'day', 'hour', 'channel']].groupby(by=['ip', 'day', 'hour'])[['channel']].count().reset_index()\
        .rename(index=str, columns={"channel": 'qty'})
    x_train = x_train.merge(gp, on=['ip', 'day', 'hour'], how='left')

    gp = x_train[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(
        index=str, columns={"channel": 'ip_app_count'})
    x_train = x_train.merge(gp, on=['ip', 'app'], how='left')

    gp = x_train[['ip', 'app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().\
        rename(index=str, columns={"channel": 'ip_app_os_count'})
    x_train = x_train.merge(gp, on=['ip', 'app', 'os'], how='left')

    gb = x_train[['device', 'os', 'channel']].groupby(by=['device', 'os'])[['channel']].count().reset_index().\
        rename(index=str, columns={"channel": "device_os_count"})
    x_train = x_train.merge(gb, on=['device', 'os'], how='left')

    x_train['qty'] = x_train['qty'].astype('uint16')
    x_train['ip_app_count'] = x_train['ip_app_count'].astype('uint16')
    x_train['ip_app_os_count'] = x_train['ip_app_os_count'].astype('uint16')
    x_train['device_os_count'] = x_train['device_os_count'].astype('unit16')

    x_val = x_train[:1000000]
    x_train = x_train[1000000:]

    return x_train, x_val


def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                 feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.01,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
    }

    lgb_params.update(params)

    print("preparing validation datasets")

    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features
                          )

    evals_results = {}

    bst1 = lgb.train(lgb_params,
                     xgtrain,
                     valid_sets=[xgtrain, xgvalid],
                     valid_names=['train', 'valid'],
                     evals_result=evals_results,
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10,
                     feval=feval)

    n_estimators = bst1.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])

    return bst1


def main():
    x_train = load()
    x_train, x_val = predispose(x_train)

    target = 'is_attributed'
    predictors = ['app', 'device', 'os', 'channel', 'hour', 'day', 'qty', 'ip_app_count', 'ip_app_os_count']
    categorical = ['app', 'device', 'os', 'channel', 'hour']
    params = {
        'learning_rate': 0.1,
        'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight': 99  # because training data is extremely unbalanced
    }

    bst = lgb_modelfit_nocv(params,
                            x_train,
                            x_val,
                            predictors,
                            target,
                            objective='binary',
                            metrics='auc',
                            early_stopping_rounds=50,
                            verbose_eval=True,
                            num_boost_round=300,
                            categorical_features=categorical)


if __name__ == '__main__':
    main()