#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 0004 下午 16:43
# @Author  : Juzphy

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')


def tpr_weight_function(y_true, y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3


def eval_error(pred, train_set):
    labels = train_set.get_label()
    score = tpr_weight_function(labels, pred)
    return 'Orange', score, True


def sub_on_line(train_, valid_, doc_path, is_shuffle=True):
    print(f'data shape:\ntrain--{train_.shape}\nvalid--{valid_.shape}')
    folds = StratifiedKFold(n_splits=5, shuffle=is_shuffle, random_state=1024)
    pred = [k for k in train_.columns if k not in ['Tag', 'UID']]
    sub_preds = np.zeros((valid_.shape[0], folds.n_splits))
    print(f'Use {len(pred)} features ...')
    res_e = []
    params = {
        'learning_rate': 0.01,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 20,
        'nthread': -1,
        'verbose': -1
    }
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_, train_['Tag']), start=1):
        print(f'the {n_fold} training start ...')
        train_x, train_y = train_[pred].iloc[train_idx], train_['Tag'].iloc[train_idx]
        valid_x, valid_y = train_[pred].iloc[valid_idx], train_['Tag'].iloc[valid_idx]

        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)

        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=10000,
            valid_sets=[dvalid],
            early_stopping_rounds=100,
            # feval=eval_error,
            verbose_eval=100
        )
        sub_preds[:, n_fold - 1] = clf.predict(valid_[pred], num_iteration=clf.best_iteration)
        train_pred = clf.predict(valid_x, num_iteration=clf.best_iteration)
        tmp_score = tpr_weight_function(valid_y, train_pred)
        res_e.append(tmp_score)
        print(f'Orange custom score: {tmp_score}')
    valid_['Tag'] = np.mean(sub_preds, axis=1)

    valid_res = pd.read_csv(open(data_path+'test_tag_r1.csv', encoding='utf8'))[['UID']]
    valid_res = valid_res.merge(valid_[['UID', 'Tag']], how='left', on='UID')
    valid_res.to_csv(doc_path, index=False)
    print(res_e)
    print('five folds average score: ', np.mean(res_e))


if __name__ == '__main__':
    data_path = './'
    train_stat = pd.read_csv(open(data_path+'gen_data/juz_train_data.csv'))
    valid_stat = pd.read_csv(open(data_path+'gen_data/juz_test_data.csv'))
    sub_on_line(train_stat, valid_stat, './result/submit_a_lgb.csv')

