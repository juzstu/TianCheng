#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/12/4 0004 下午 16:43
# @Author  : Juzphy

import pandas as pd
import numpy as np
from itertools import combinations
import os
import geohash
import warnings

warnings.filterwarnings('ignore')


# 获取设备众数
def mode_count(x):
    try:
        return x.value_counts().iloc[0]
    except Exception:
        return np.nan


def data_group(data, col):
    # day
    df = pd.DataFrame(data.groupby(col)['UID'].nunique())
    df.columns = ['cnt_uid_' + col]
    df['early_day_' + col] = data.groupby(col)['day2'].min()
    df['later_day_' + col] = data.groupby(col)['day2'].max()
    df['range_day_' + col] = df['later_day_' + col] - df['early_day_' + col]
    if col != 'geo_code':
        # longitude
        df['close_long_' + col] = data.groupby(col)['longitude'].min()
        df['far_long_' + col] = data.groupby(col)['longitude'].max()
        df['range_long_' + col] = df['far_long_' + col] - df['close_long_' + col]
        # latitude
        df['close_lat_' + col] = data.groupby(col)['latitude'].min()
        df['far_lat_' + col] = data.groupby(col)['latitude'].max()
        df['range_lat_' + col] = df['far_lat_' + col] - df['close_lat_' + col]

    df['least_ratio_' + col] = data.groupby(col)['ratio'].min()
    df['most_ratio_' + col] = data.groupby(col)['ratio'].max()
    df['range_ratio_' + col] = df['most_ratio_' + col] - df['least_ratio_' + col]

    if col not in ['device_code1', 'device_code2', 'device_code3']:
        df['dc1_' + col] = data.groupby(col)['device_code1'].apply(mode_count)
        df['dc2_' + col] = data.groupby(col)['device_code2'].apply(mode_count)
        df['dc3_' + col] = data.groupby(col)['device_code3'].apply(mode_count)

    df = df.reset_index()
    return df


def data_group_pair(data, col):
    col2 = '_'.join(col)
    df = pd.DataFrame(data.groupby(col)['UID'].nunique())
    df.columns = ['cnt_uid_' + col2]
    df['early_day_' + col2] = data.groupby(col)['day2'].min()
    df['later_day_' + col2] = data.groupby(col)['day2'].max()
    df['range_day_' + col2] = df['later_day_' + col2] - df['early_day_' + col2]

    df = df.reset_index()
    return df


def extract_feat(data_frame):
    # UID类的特征
    data_tmp = pd.DataFrame(data_frame.groupby('UID')['day'].min())
    data_tmp.columns = ['uid_early_day']
    data_tmp['uid_max_day'] = data_frame.groupby('UID')['day'].max()
    data_tmp['uid_range_day'] = data_tmp['uid_max_day'] - data_tmp['uid_early_day']

    data_tmp['uid_min_long'] = data_frame.groupby('UID')['longitude'].min()
    data_tmp['uid_max_long'] = data_frame.groupby('UID')['longitude'].max()
    data_tmp['uid_range_long'] = data_tmp['uid_max_long'] - data_tmp['uid_min_long']

    data_tmp['uid_min_lat'] = data_frame.groupby('UID')['latitude'].min()
    data_tmp['uid_max_lat'] = data_frame.groupby('UID')['latitude'].max()
    data_tmp['uid_range_lat'] = data_tmp['uid_max_lat'] - data_tmp['uid_min_lat']

    data_tmp['uid_min_amt'] = data_frame.groupby('UID')['trans_amt'].min()
    data_tmp['uid_max_amt'] = data_frame.groupby('UID')['trans_amt'].max()
    data_tmp['uid_range_amt'] = data_tmp['uid_max_amt'] - data_tmp['uid_min_amt']

    data_tmp['uid_min_bal'] = data_frame.groupby('UID')['bal'].min()
    data_tmp['uid_max_bal'] = data_frame.groupby('UID')['bal'].max()
    data_tmp['uid_range_bal'] = data_tmp['uid_max_bal'] - data_tmp['uid_min_bal']

    for wd in range(7):
        data_tmp[f'uid_week_{wd}_cnt'] = data_frame[data_frame['nweek_day'] == wd].groupby('UID')['day'].count()
    data_tmp['uid_all_cnt'] = data_frame.groupby('UID')['day'].count()

    time_period = [-1, 8, 12, 15, 24]
    for tp in range(4):
        data_tmp[f'uid_time_period_{tp}_cnt'] = \
            data_frame[((data_frame['nhour'] > time_period[tp]) &
                        (data_frame['nhour'] <= time_period[tp+1]))].groupby('UID')['day'].count()

    # UID关联的自身属性
    nunique_var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'bal', 'channel', 'code1', 'code2', 'day',
                   'device1', 'device2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub',
                   'ip2', 'ip2_sub', 'mac1', 'mac2', 'market_code', 'market_type', 'merchant', 'mode', 'nweek_day',
                   'os', 'success', 'trans_amt', 'trans_type1', 'trans_type2', 'version', 'wifi', 'nhour', 'longitude'
                   , 'latitude']

    for nv in nunique_var:
        data_tmp[f'uid_{nv}_nunique'] = data_frame.groupby('UID')[nv].nunique()
        data_tmp[f'uid_{nv}_notnull'] = data_frame[data_frame[nv].notnull()].groupby('UID')['UID'].count()

    data_tmp.drop(['uid_nhour_notnull', 'uid_day_notnull', 'uid_nweek_day_notnull'], axis=1, inplace=True)
    data_tmp['uid_tran_cnt'] = data_frame[data_frame['mode'] == 'transaction'].groupby('UID')['UID'].count()

    # UID关联到的设备、ip、位置的信息

    relate_var = ['acc_id1', 'acc_id2', 'acc_id3', 'amt_src1', 'amt_src2', 'version', 'code1', 'code2',
                  'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 'ip1_sub', 'ip2',
                  'ip2_sub', 'mac1', 'mac2', 'market_code', 'merchant', 'wifi', 'mode']

    relate_pair = ['code1', 'code2', 'device_code1', 'device_code2', 'device_code3', 'geo_code', 'ip1', 
                   'ip1_sub', 'ip2', 'ip2_sub', 'mac1', 'mac2', 'merchant', 'wifi', 'mode']

    for rv in relate_var:
        print(f'waiting for generating feature of {rv} ...')
        sample_data = data_frame[['UID', rv]].drop_duplicates()
        group_data = data_group(data_frame, rv)
        sample_data = sample_data.merge(group_data, on=rv, how='left')
        data_tmp['relate_cnt_uid_' + rv + '_max'] = sample_data.groupby('UID')['cnt_uid_' + rv].max()
        data_tmp['relate_cnt_uid_' + rv + '_min'] = sample_data.groupby('UID')['cnt_uid_' + rv].min()
        data_tmp['relate_cnt_uid_' + rv + '_mean'] = sample_data.groupby('UID')['cnt_uid_' + rv].mean()
        data_tmp['relate_cnt_uid_' + rv + '_skew'] = sample_data.groupby('UID')['cnt_uid_' + rv].skew()

        if rv not in ['device_code1', 'device_code2', 'device_code3']:
            data_tmp['relate_dc1_' + rv + '_max'] = sample_data.groupby('UID')['dc1_' + rv].max()
            data_tmp['relate_dc1_' + rv + '_min'] = sample_data.groupby('UID')['dc1_' + rv].min()
            data_tmp['relate_dc1_' + rv + '_mean'] = sample_data.groupby('UID')['dc1_' + rv].mean()
            data_tmp['relate_dc1_' + rv + '_skew'] = sample_data.groupby('UID')['dc1_' + rv].skew()

            data_tmp['relate_dc2_' + rv + '_max'] = sample_data.groupby('UID')['dc2_' + rv].max()
            data_tmp['relate_dc2_' + rv + '_min'] = sample_data.groupby('UID')['dc2_' + rv].min()
            data_tmp['relate_dc2_' + rv + '_mean'] = sample_data.groupby('UID')['dc2_' + rv].mean()
            data_tmp['relate_dc2_' + rv + '_skew'] = sample_data.groupby('UID')['dc2_' + rv].skew()

            data_tmp['relate_dc3_' + rv + '_max'] = sample_data.groupby('UID')['dc3_' + rv].max()
            data_tmp['relate_dc3_' + rv + '_min'] = sample_data.groupby('UID')['dc3_' + rv].min()
            data_tmp['relate_dc3_' + rv + '_mean'] = sample_data.groupby('UID')['dc3_' + rv].mean()
            data_tmp['relate_dc3_' + rv + '_skew'] = sample_data.groupby('UID')['dc3_' + rv].skew()

        data_tmp['relate_early_day_' + rv + '_max'] = sample_data.groupby('UID')['early_day_' + rv].max()
        data_tmp['relate_early_day_' + rv + '_min'] = sample_data.groupby('UID')['early_day_' + rv].min()
        data_tmp['relate_early_day_' + rv + '_mean'] = sample_data.groupby('UID')['early_day_' + rv].mean()
        data_tmp['relate_early_day_' + rv + '_skew'] = sample_data.groupby('UID')['early_day_' + rv].skew()

        data_tmp['relate_later_day_' + rv + '_max'] = sample_data.groupby('UID')['later_day_' + rv].max()
        data_tmp['relate_later_day_' + rv + '_min'] = sample_data.groupby('UID')['later_day_' + rv].min()
        data_tmp['relate_later_day_' + rv + '_mean'] = sample_data.groupby('UID')['later_day_' + rv].mean()
        data_tmp['relate_later_day_' + rv + '_skew'] = sample_data.groupby('UID')['later_day_' + rv].skew()

        data_tmp['relate_range_day_' + rv + '_max'] = sample_data.groupby('UID')['range_day_' + rv].max()
        data_tmp['relate_range_day_' + rv + '_min'] = sample_data.groupby('UID')['range_day_' + rv].min()
        data_tmp['relate_range_day_' + rv + '_mean'] = sample_data.groupby('UID')['range_day_' + rv].mean()
        data_tmp['relate_range_day_' + rv + '_skew'] = sample_data.groupby('UID')['range_day_' + rv].skew()

        if rv != 'geo_code':
            data_tmp['relate_close_long_' + rv + '_max'] = sample_data.groupby('UID')['close_long_' + rv].max()
            data_tmp['relate_close_long_' + rv + '_min'] = sample_data.groupby('UID')['close_long_' + rv].min()
            data_tmp['relate_close_long_' + rv + '_mean'] = sample_data.groupby('UID')['close_long_' + rv].mean()
            data_tmp['relate_close_long_' + rv + '_skew'] = sample_data.groupby('UID')['close_long_' + rv].skew()

            data_tmp['relate_far_long_' + rv + '_max'] = sample_data.groupby('UID')['far_long_' + rv].max()
            data_tmp['relate_far_long_' + rv + '_min'] = sample_data.groupby('UID')['far_long_' + rv].min()
            data_tmp['relate_far_long_' + rv + '_mean'] = sample_data.groupby('UID')['far_long_' + rv].mean()
            data_tmp['relate_far_long_' + rv + '_skew'] = sample_data.groupby('UID')['far_long_' + rv].skew()

            data_tmp['relate_range_long_' + rv + '_max'] = sample_data.groupby('UID')['range_long_' + rv].max()
            data_tmp['relate_range_long_' + rv + '_min'] = sample_data.groupby('UID')['range_long_' + rv].min()
            data_tmp['relate_range_long_' + rv + '_mean'] = sample_data.groupby('UID')['range_long_' + rv].mean()
            data_tmp['relate_range_long_' + rv + '_skew'] = sample_data.groupby('UID')['range_long_' + rv].skew()

            data_tmp['relate_close_lat_' + rv + '_max'] = sample_data.groupby('UID')['close_lat_' + rv].max()
            data_tmp['relate_close_lat_' + rv + '_min'] = sample_data.groupby('UID')['close_lat_' + rv].min()
            data_tmp['relate_close_lat_' + rv + '_mean'] = sample_data.groupby('UID')['close_lat_' + rv].mean()
            data_tmp['relate_close_lat_' + rv + '_skew'] = sample_data.groupby('UID')['close_lat_' + rv].skew()

            data_tmp['relate_far_lat_' + rv + '_max'] = sample_data.groupby('UID')['far_lat_' + rv].max()
            data_tmp['relate_far_lat_' + rv + '_min'] = sample_data.groupby('UID')['far_lat_' + rv].min()
            data_tmp['relate_far_lat_' + rv + '_mean'] = sample_data.groupby('UID')['far_lat_' + rv].mean()
            data_tmp['relate_far_lat_' + rv + '_skew'] = sample_data.groupby('UID')['far_lat_' + rv].skew()

            data_tmp['relate_range_lat_' + rv + '_max'] = sample_data.groupby('UID')['range_lat_' + rv].max()
            data_tmp['relate_range_lat_' + rv + '_min'] = sample_data.groupby('UID')['range_lat_' + rv].min()
            data_tmp['relate_range_lat_' + rv + '_mean'] = sample_data.groupby('UID')['range_lat_' + rv].mean()
            data_tmp['relate_range_lat_' + rv + '_skew'] = sample_data.groupby('UID')['range_lat_' + rv].skew()

        data_tmp['relate_least_ratio_' + rv + '_max'] = sample_data.groupby('UID')['least_ratio_' + rv].max()
        data_tmp['relate_least_ratio_' + rv + '_min'] = sample_data.groupby('UID')['least_ratio_' + rv].min()
        data_tmp['relate_least_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['least_ratio_' + rv].mean()
        data_tmp['relate_least_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['least_ratio_' + rv].skew()

        data_tmp['relate_most_ratio_' + rv + '_max'] = sample_data.groupby('UID')['most_ratio_' + rv].max()
        data_tmp['relate_most_ratio_' + rv + '_min'] = sample_data.groupby('UID')['most_ratio_' + rv].min()
        data_tmp['relate_most_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['most_ratio_' + rv].mean()
        data_tmp['relate_most_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['most_ratio_' + rv].skew()

        data_tmp['relate_range_ratio_' + rv + '_max'] = sample_data.groupby('UID')['range_ratio_' + rv].max()
        data_tmp['relate_range_ratio_' + rv + '_min'] = sample_data.groupby('UID')['range_ratio_' + rv].min()
        data_tmp['relate_range_ratio_' + rv + '_mean'] = sample_data.groupby('UID')['range_ratio_' + rv].mean()
        data_tmp['relate_range_ratio_' + rv + '_skew'] = sample_data.groupby('UID')['range_ratio_' + rv].skew()

    for rv in combinations(relate_pair, 2):
        print(f'waiting for group pair features of {rv} ...')
        rv2 = '_'.join(rv)
        sample_data = data_frame[['UID'] + list(rv)].drop_duplicates()
        group_data = data_group_pair(data_frame, rv)
        sample_data = sample_data.merge(group_data, on=rv, how='left')

        data_tmp['relate_cnt_uid_' + rv2 + '_max'] = sample_data.groupby('UID')['cnt_uid_' + rv2].max()
        data_tmp['relate_cnt_uid_' + rv2 + '_min'] = sample_data.groupby('UID')['cnt_uid_' + rv2].min()
        data_tmp['relate_cnt_uid_' + rv2 + '_mean'] = sample_data.groupby('UID')['cnt_uid_' + rv2].mean()
        data_tmp['relate_cnt_uid_' + rv2 + '_skew'] = sample_data.groupby('UID')['cnt_uid_' + rv2].skew()

        data_tmp['relate_early_day_' + rv2 + '_max'] = sample_data.groupby('UID')['early_day_' + rv2].max()
        data_tmp['relate_early_day_' + rv2 + '_min'] = sample_data.groupby('UID')['early_day_' + rv2].min()
        data_tmp['relate_early_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['early_day_' + rv2].mean()
        data_tmp['relate_early_day_' + rv2 + '_skew'] = sample_data.groupby('UID')['early_day_' + rv2].skew()

        data_tmp['relate_later_day_' + rv2 + '_max'] = sample_data.groupby('UID')['later_day_' + rv2].max()
        data_tmp['relate_later_day_' + rv2 + '_min'] = sample_data.groupby('UID')['later_day_' + rv2].min()
        data_tmp['relate_later_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['later_day_' + rv2].mean()
        data_tmp['relate_later_day_' + rv2 + '_skew'] = sample_data.groupby('UID')['later_day_' + rv2].skew()

        data_tmp['relate_range_day_' + rv2 + '_max'] = sample_data.groupby('UID')['range_day_' + rv2].max()
        data_tmp['relate_range_day_' + rv2 + '_min'] = sample_data.groupby('UID')['range_day_' + rv2].min()
        data_tmp['relate_range_day_' + rv2 + '_mean'] = sample_data.groupby('UID')['range_day_' + rv2].mean()
        data_tmp['relate_range_day_' + rv2 + '_skew'] = sample_data.groupby('UID')['range_day_' + rv2].skew()

    return data_tmp


if __name__ == '__main__':
    data_path = './'
    operation_train = pd.read_csv(open(data_path+'operation_train_new.csv', encoding='utf8'))
    transaction_train = pd.read_csv(open(data_path+'transaction_train_new.csv', encoding='utf8'))
    tag_train = pd.read_csv(open(data_path+'tag_train_new.csv', encoding='utf8'))

    operation_round1 = pd.read_csv(open(data_path+'operation_round1_new.csv', encoding='utf8'))
    transaction_round1 = pd.read_csv(open(data_path+'transaction_round1_new.csv', encoding='utf8'))
    tag_valid = pd.read_csv(open(data_path+'test_tag_r1.csv', encoding='utf8'))[['UID']]
    print('Successed load in train and test data.')
    
    transaction_train['mode'] = 'transaction'
    action_train = operation_train.append(transaction_train).reset_index(drop=True)
    action_train = action_train.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])
    action_train = action_train.merge(tag_train, on='UID')
    action_train['day2'] = action_train['day']

    transaction_round1['mode'] = 'transaction'
    action_round1 = operation_round1.append(transaction_round1).reset_index(drop=True)
    action_round1 = action_round1.sort_values(by=['UID', 'day', 'time'], ascending=[True, True, True])
    action_round1['day2'] = action_round1['day'] + 30

    all_data = action_train.append(action_round1).reset_index(drop=True)
    all_data['nweek_day'] = all_data['day'].apply(lambda x: x % 7)
    all_data['version'] = all_data.version.fillna('0.0.0')
    all_data['nhour'] = all_data['time'].apply(lambda x: int(x[:2]))
    all_data['longitude'] = all_data['geo_code'].apply(lambda x: geohash.decode(x)[0] if isinstance(x, str) else np.nan)
    all_data['latitude'] = all_data['geo_code'].apply(lambda x: geohash.decode(x)[1] if isinstance(x, str) else np.nan)
    all_data['ratio'] = all_data['trans_amt'] / all_data['bal']

    data_var = extract_feat(all_data)
    data_var = data_var.reset_index()

    train = tag_train.merge(data_var, on='UID')
    valid = tag_valid.merge(data_var, on='UID')
    print(f'Gen train shape: {train.shape}, test shape: {valid.shape}')

    drop_train = train.T.drop_duplicates().T
    drop_valid = valid.T.drop_duplicates().T
    
    features = [i for i in drop_train.columns if i in drop_valid.columns]
    print('features num: ', len(features)-1)
    if not os.path.exists('gen_data'):
        os.mkdir('gen_data')
    train[features + ['Tag']].to_csv('./gen_data/juz_train_data.csv', index=False)
    valid[features].to_csv('./gen_data/juz_test_data.csv', index=False)


