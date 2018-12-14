#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 0004 下午 16:43
# @Author  : Juzphy

import pandas as pd
import warnings
from gensim.models import Word2Vec
import multiprocessing

warnings.filterwarnings('ignore')


def w2v_feat(data_frame, feat, mode):
    for i in feat:
        if data_frame[i].dtype != 'object':
            data_frame[i] = data_frame[i].astype(str)
    data_frame.fillna('nan', inplace=True)

    print(f'Start {mode} word2vec ...')
    model = Word2Vec(data_frame[feat].values.tolist(), size=L, window=2, min_count=1,
                     workers=multiprocessing.cpu_count(), iter=10)
    stat_list = ['min', 'max', 'mean', 'std']
    new_all = pd.DataFrame()
    for m, t in enumerate(feat):
        print(f'Start gen feat of {t} ...')
        tmp = []
        for i in data_frame[t].unique():
            tmp_v = [i]
            tmp_v.extend(model[i])
            tmp.append(tmp_v)
        tmp_df = pd.DataFrame(tmp)
        w2c_list = [f'w2c_{t}_{n}' for n in range(L)]
        tmp_df.columns = [t] + w2c_list
        tmp_df = data_frame[['UID', t]].merge(tmp_df, on=t)
        tmp_df = tmp_df.drop_duplicates().groupby('UID').agg(stat_list).reset_index()
        tmp_df.columns = ['UID'] + [f'{p}_{q}' for p in w2c_list for q in stat_list]
        if m == 0:
            new_all = pd.concat([new_all, tmp_df], axis=1)
        else:
            new_all = pd.merge(new_all, tmp_df, how='left', on='UID')
    return new_all


if __name__ == '__main__':
    L = 10
    data_path = './'
    operation_train = pd.read_csv(open(data_path+'operation_train_new.csv', encoding='utf8'))
    transaction_train = pd.read_csv(open(data_path+'transaction_train_new.csv', encoding='utf8'))
    tag_train = pd.read_csv(open(data_path+'tag_train_new.csv', encoding='utf8'))

    operation_round1 = pd.read_csv(open(data_path+'test_operation_round1.csv', encoding='utf8'))
    transaction_round1 = pd.read_csv(open(data_path+'test_transaction_round1.csv', encoding='utf8'))
    tag_b = pd.read_csv(open(data_path+'test_tag_r1.csv', encoding='utf8'))[['UID']]
    print('Successed load in train and test data.')

    transaction_train['mode'] = 'transaction'
    action_train = operation_train.append(transaction_train).reset_index(drop=True)
    action_train = action_train.merge(tag_train, on='UID')

    transaction_round1['mode'] = 'transaction'
    action_round1 = operation_round1.append(transaction_round1).reset_index(drop=True)

    trans_feat = ['day', 'time', 'geo_code', 'device1', 'device2', 'device_code1', 'device_code2', 'device_code3',
                  'mac1', 'wifi', 'mac2', 'ip1', 'ip2', 'merchant', 'code1', 'code2', 'acc_id1', 'acc_id2', 'acc_id3',
                  'version', 'channel', 'market_code', 'market_type', 'trans_type1', 'trans_type2', 'trans_amt', 'bal',
                  'amt_src1', 'amt_src2', 'mode']

    new_all_train = w2v_feat(action_train, trans_feat, 'train')
    new_all_test = w2v_feat(action_round1, trans_feat, 'test')
    train = pd.merge(tag_train, new_all_train, on='UID', how='left')
    valid = pd.merge(tag_b, new_all_test, on='UID', how='left')
    print(f'Gen train shape: {train.shape}, test shape: {valid.shape}')

    drop_train = train.T.drop_duplicates().T
    drop_valid = valid.T.drop_duplicates().T

    features = [i for i in drop_train.columns if i in drop_valid.columns]
    print('features num: ', len(features) - 1)
    train[features + ['Tag']].to_csv('./gen_data/juz_train_w2v.csv', index=False)
    valid[features].to_csv('./gen_data/juz_test_w2v.csv', index=False)
