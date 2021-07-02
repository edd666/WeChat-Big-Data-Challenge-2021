# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    特征
"""

# packages
import numpy as np
from config import *
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler


def extract_features(df, action):
    """
    提取特征

    :param df: DataFrame 样本(训练和测试)
    :param action: str action
    :return:
        x: DataFrame 特征
        y: Series 标签
    """
    # 特征-训练每个任务都需要初始化
    DENSE_FEATURE_COLUMNS = ['videoplayseconds']

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

    feature_columns = DENSE_FEATURE_COLUMNS + SPARSE_FEATURE_COLUMNS + \
                      VARLEN_SPARSE_FEATURE_COLUMNS + list(WEIGHT_NAME.values())

    train_x, train_y = train_df[feature_columns], train_df[action]

    test_x, test_y = test_df[feature_columns], test_df[action]

    return train_x, train_y, test_x, test_y
