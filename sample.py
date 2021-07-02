# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    样本生成
"""

# packages
import pandas as pd
from os.path import exists
from config import *


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

