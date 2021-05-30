# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-05-25
# @Contact : liaozhi_edo@163.com


"""
    样本和特征的拼接
"""

# packages
import numpy as np
import pandas as pd
from os.path import join
from config import DATA_DIR, SAVE_DIR, ACTION_LIST, STAGE_END_DAY, FEATURE_COLUMNS, SPARSE_FEATURE_COLUMNS


def sample_feature_concat(sample_df_list, stage='offline_train'):
    """
    样本和特征的拼接

    :param sample_df_list: list of DataFrame 样本
    :param stage: str stage
    :return:
    """
    # 1,加载特征
    feature_dir = join(SAVE_DIR, 'feature')
    # user feature
    user_feature_path = join(feature_dir, 'user_feature.csv')
    user_feature_df = pd.read_csv(user_feature_path, header=0, index_col=False)
    # feed feature
    feed_feature_path = join(feature_dir, 'feed_feature.csv')
    feed_feature_df = pd.read_csv(feed_feature_path, header=0, index_col=False)
    # feed info
    columns = ['feedid', 'authorid', 'videoplayseconds', 'bgm_song_id', 'bgm_singer_id']
    info_df = pd.read_csv(DATA_DIR + 'feed_info.csv', header=0, index_col=False, usecols=columns)
    info_df[['authorid', 'bgm_song_id', 'bgm_singer_id']] += 1  # 0 用于填未知
    values = {
        'authorid': 0,
        'bgm_song_id': 0,
        'bgm_singer_id': 0,
        'videoplayseconds': 0,
    }
    info_df.fillna(value=values, inplace=True)
    dtypes = {
        'authorid': 'int',
        'bgm_song_id': 'int',
        'bgm_singer_id': 'int',
    }
    info_df = info_df.astype(dtype=dtypes, copy=True)

    # 2,样本和特征的拼接
    day = STAGE_END_DAY[stage]
    stage_dir = join(SAVE_DIR, stage)
    for idx, sample_df in enumerate(sample_df_list):
        if stage == 'submit':
            action = 'all'
            feature_label_columns = FEATURE_COLUMNS
        elif stage == 'evaluate':
            action = 'all'
            feature_label_columns = FEATURE_COLUMNS + ACTION_LIST
        else:
            action = ACTION_LIST[idx]
            feature_label_columns = FEATURE_COLUMNS + [action]

        # 拼接
        sample_feature_df = sample_df.merge(info_df, how='left', on=['feedid'], suffixes=(None, '_feed'))
        sample_feature_df = sample_feature_df.merge(feed_feature_df, on=['feedid', 'date_'],
                                                    how='left', suffixes=(None, '_feed'))
        sample_feature_df = sample_feature_df.merge(user_feature_df, on=['userid', 'date_'],
                                                    how='left', suffixes=('_feed', '_user'))
        # 缺失值处理(此处只有连续值特征存在NAN)
        sample_feature_df.fillna(value=0, inplace=True)

        # 连续值平滑
        dense_feature_columns = list(set(FEATURE_COLUMNS) - set(SPARSE_FEATURE_COLUMNS))
        sample_feature_df[dense_feature_columns] = np.log(1.0 + sample_feature_df[dense_feature_columns])

        # 保存
        file_path = join(stage_dir, stage + '_' + action + '_' + str(day) + '_' + 'sample_feature_concat.csv')
        sample_feature_df[feature_label_columns].to_csv(file_path, header=True, index=False)
        print('Save to: %s' % file_path)

    return

