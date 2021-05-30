# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-05-25
# @Contact : liaozhi_edo@163.com


"""
    特征
"""

# packages
import pandas as pd
from os.path import join
from config import DATA_DIR, SAVE_DIR, END_DAY


def user_features(action_df):
    """
    构建用户特征

    :param action_df: DataFrame 行为日志
    :return:
        user_feature_df: DataFrame 用户特征
    """
    # 1,统计特征(PV)
    stat_df = action_df.groupby(by=['userid']).agg(func={
        'date_': 'nunique',
        'feedid': 'count',
        'read_comment': 'sum',
        'like': 'sum',
        'click_avatar': 'sum',
        'forward': 'sum',
        'finish': 'sum',
        'play': 'mean',
        'stay': 'mean',
        'interact': 'mean',
        'no_play': 'mean',
        'no_stay': 'mean',
        'videoplayseconds': 'mean',
    })
    stat_df.rename(columns={
        'date_': 'login_day_count',
        'feedid': 'browse_count',
        'read_comment': 'read_comment_count',
        'like': 'like_count',
        'click_avatar': 'click_avatar_count',
        'forward': 'forward_count',
        'finish': 'finish_count',
        'play': 'avg_play',
        'stay': 'avg_stay',
        'interact': 'avg_interact',
        'no_play': 'avg_no_play',
        'no_stay': 'avg_no_stay',
        'videoplayseconds': 'avg_videoplayseconds',  # 用户对视频时长的偏好
    }, inplace=True)

    # 计算比例
    stat_df['read_comment_rate'] = stat_df['read_comment_count'] / stat_df['browse_count']
    stat_df['like_rate'] = stat_df['like_count'] / stat_df['browse_count']
    stat_df['click_avatar_rate'] = stat_df['click_avatar_count'] / stat_df['browse_count']
    stat_df['forward_rate'] = stat_df['forward_count'] / stat_df['browse_count']
    stat_df['finish_rate'] = stat_df['finish_count'] / stat_df['browse_count']
    stat_df.fillna(value=0.0, inplace=True)  # 填充缺失值

    # 2,特征合并
    list_df_to_concat = [stat_df]
    user_feature_df = pd.concat(list_df_to_concat, axis=1, join='outer').reset_index()

    return user_feature_df


def feed_features(action_df):
    """
    构建视频流(feed)特征

    :param action_df: DataFrame 行为日志
    :return:
        feed_feature_df: DataFrame feed流特征
    """
    # 1,统计特征(PV)
    stat_df = action_df.groupby(by=['feedid']).agg(func={
        'userid': 'count',
        'read_comment': 'sum',
        'like': 'sum',
        'click_avatar': 'sum',
        'forward': 'sum',
        'finish': 'sum',
        'play': 'mean',
        'stay': 'mean',
        'interact': 'mean',
        'no_play': 'mean',
        'no_stay': 'mean',
    })
    stat_df.rename(columns={
        'userid': 'show_count',
        'read_comment': 'read_comment_count',
        'like': 'like_count',
        'click_avatar': 'click_avatar_count',
        'forward': 'forward_count',
        'finish': 'finish_count',
        'play': 'avg_play',
        'stay': 'avg_stay',
        'interact': 'avg_interact',
        'no_play': 'avg_no_play',
        'no_stay': 'avg_no_stay',
    }, inplace=True)

    # 计算比例
    stat_df['read_comment_rate'] = stat_df['read_comment_count'] / stat_df['show_count']
    stat_df['like_rate'] = stat_df['like_count'] / stat_df['show_count']
    stat_df['click_avatar_rate'] = stat_df['click_avatar_count'] / stat_df['show_count']
    stat_df['forward_rate'] = stat_df['forward_count'] / stat_df['show_count']
    stat_df['finish_rate'] = stat_df['finish_count'] / stat_df['show_count']
    stat_df.fillna(value=0.0, inplace=True)  # 填充缺失值

    # 2,特征合并
    list_df_to_concat = [stat_df]
    # join='inner',只匹配行为记录中出现过的feedid
    feed_feature_df = pd.concat(list_df_to_concat, axis=1, join='outer').reset_index()

    return feed_feature_df


def construct_features(start_day=1, before_days=7):
    """
    利用滑窗法构建特征

    注意: start + before_days为样本时间

    :param start_day: str 构建特征时数据选取的初始时间
    :param before_days: int 选择样本时间之前多少天数据来构建特征
    :return:
    """
    # 1,加载数据
    action_df = pd.read_csv(DATA_DIR + 'user_action.csv', header=0, index_col=False)
    columns = ['feedid', 'videoplayseconds']  # 读入特定列以减少内存使用
    feed_df = pd.read_csv(DATA_DIR + 'feed_info.csv', header=0, index_col=False, usecols=columns)
    action_df = action_df.merge(feed_df, on=['feedid'], how='left')

    # 2,数据处理
    action_df['play'] = action_df['play'] / 1000
    action_df['stay'] = action_df['stay'] / 1000
    # no_play/no_stay分别表示视频时长-play/stay
    action_df['no_play'] = action_df['videoplayseconds'] - action_df['play']
    action_df['no_stay'] = action_df['videoplayseconds'] - action_df['stay']
    action_df['interact'] = action_df[['stay', 'play']].apply(
        lambda S: S['stay'] - S['play'] if S['stay'] - S['play'] >= 0 else 0, axis=1)
    action_df['finish'] = action_df[['play', 'videoplayseconds']].apply(
        lambda S: 1 if S['play'] - S['videoplayseconds'] >= 0 else 0, axis=1)

    # 3,利用滑窗法构建特征
    user_feature_df_list = []
    feed_feature_df_list = []
    for start in range(start_day, END_DAY - before_days + 1):
        # 数据范围: start : start + before_days - 1
        # 样本时间: start + before_days
        print('Data range: %s, Sample date: %s' % ((start, start + before_days - 1), start + before_days))
        tmp_df = action_df.loc[(action_df.date_ >= start) & (action_df.date_ <= start + before_days - 1), :]

        # user feature
        user_feature_df = user_features(tmp_df)
        user_feature_df['date_'] = start + before_days
        user_feature_df_list.append(user_feature_df)

        # feed feature
        feed_feature_df = feed_features(tmp_df)
        feed_feature_df['date_'] = start + before_days
        feed_feature_df_list.append(feed_feature_df)

    # 4,合并特征并保存
    # user
    feature_dir = join(SAVE_DIR, 'feature')
    user_feature_df = pd.concat(user_feature_df_list, ignore_index=True, sort=True)
    user_feature_path = join(feature_dir, 'user_feature.csv')
    user_feature_df.to_csv(user_feature_path, header=True, index=False)
    print('Save to: %s' % user_feature_path)

    # feed
    feed_feature_df = pd.concat(feed_feature_df_list, ignore_index=True, sort=True)
    feed_feature_path = join(feature_dir, 'feed_feature.csv')
    feed_feature_df.to_csv(feed_feature_path, header=True, index=False)
    print('Save to: %s' % feed_feature_path)

    return




