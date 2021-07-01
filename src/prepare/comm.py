# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    样本生成
"""

# packages
import os
import pandas as pd
import numpy as np
from os.path import exists
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from config.conf import *


def create_dir():
    """
    创建相关目录

    :return:
    """
    # 1,创建相关目录
    if not exists(SAVE_HOME):
        print('Create dir: %s' % SAVE_HOME)
        os.mkdir(SAVE_HOME)
    # 创建子目录
    need_dirs = [
        'feature', 'model', 'submit'
    ] + [
        'model/' + action
        for action in ACTION_LIST
    ]
    for need_dir in need_dirs:
        need_dir = join(SAVE_HOME, need_dir)
        if not exists(need_dir):
            print('Create dir: %s' % need_dir)
            os.mkdir(need_dir)
    return


def generate_samples():
    """
    样本生成

    :return:
        sample_df: DataFrame 所有action的训练样本和测试样本
    """
    print('\nGenerate samples')

    # 1,合并训练集和测试集
    columns = ['userid', 'feedid', 'date_', 'device', 'play', 'stay'] + ACTION_LIST
    user_action = pd.read_csv(DATA_HOME + 'user_action.csv', header=0, index_col=False, usecols=columns)
    # 为测试集添加部分列
    test = pd.read_csv(DATA_HOME + 'test_a.csv', header=0, index_col=False)
    test['date_'] = int(STAGE_END_DAY['test'])
    test['play'] = -1
    test['stay'] = -1
    for action in ACTION_LIST:
        test[action] = -1
    test = test[user_action.columns]
    # concat
    sample_df = pd.concat([user_action, test], ignore_index=False, sort=False)

    # 2,合并feed相关特征
    columns = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
    feed_info = pd.read_csv(DATA_HOME + 'feed_info.csv', header=0, index_col=False, usecols=columns)
    sample_df = sample_df.merge(feed_info, on=['feedid'], how='left')

    # 3,数据处理
    sample_df['feed'] = sample_df['feedid']

    return sample_df


def extract_features(df, action):
    """
    提取特征

    :param df: DataFrame 样本(训练和测试)
    :param action: str action
    :return:
        x: DataFrame 特征
        y: Series 标签
    """
    # 1,特征处理
    # dense
    df.fillna(value={f: 0.0 for f in DENSE_FEATURE_COLUMNS}, inplace=True)
    df[DENSE_FEATURE_COLUMNS] = np.log(1.0 + df[DENSE_FEATURE_COLUMNS])  # 平滑
    mms = MinMaxScaler(feature_range=(0, 1))
    df[DENSE_FEATURE_COLUMNS] = mms.fit_transform(df[DENSE_FEATURE_COLUMNS])  # 归一化

    # one-hot也是dense的一种,只是不需要进行平滑和归一化
    for col in ONE_HOT_COLUMNS:
        df[col] += 1
        df.fillna(value={col: 0}, inplace=True)
        encoder = OneHotEncoder(sparse=False)
        tmp = encoder.fit_transform(df[[col]])
        for idx in range(tmp.shape[1]):
            DENSE_FEATURE_COLUMNS.append(str(col) + '_' + str(idx))
            df[str(col) + '_' + str(idx)] = tmp[:, idx]

    # 数据类型转化
    df[DENSE_FEATURE_COLUMNS] = df[DENSE_FEATURE_COLUMNS].astype('float32')

    # varlen sparse
    df = df.merge(FEED_TAG, on=['feedid'], how='left')
    df = df.merge(FEED_KEYWORD, on=['feedid'], how='left')

    # sparse
    for col in SPARSE_FEATURE_COLUMNS:
        if col == 'userid':
            pass
        elif col == 'feedid':
            df[col] = df[col].apply(lambda x: FEEDID_MAP.get(x, 0))
        elif col == 'feed':
            df[col] = df[col].apply(lambda x: FEED_MAP.get(x, 0))
        elif col == 'authorid':
            pass
        else:
            df[col] += 1  # 0 用于填未知
            df.fillna(value={col: 0}, inplace=True)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])

    # 2,格式化输出
    day = STAGE_END_DAY['test']
    train_df, test_df = df.loc[df.date_ != day, :], df.loc[df.date_ == day, :]

    feature_columns = DENSE_FEATURE_COLUMNS + SPARSE_FEATURE_COLUMNS \
                      + VARLEN_SPARSE_FEATURE_COLUMNS + list(WEIGHT_NAME.values())

    train_x, train_y = train_df[feature_columns], train_df[action]

    test_x, test_y = test_df[feature_columns], test_df[action]

    return train_x, train_y, test_x, test_y


