# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-05-26
# @Contact : liaozhi_edo@163.com


"""
    配置文件
"""

# General Setting
DATA_DIR = '/data_share/weixin/wechat_algo_data1/'
SAVE_DIR = '/data_share/weixin/lz/'

# 预估行为
ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward']  # 初赛
# ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward', 'favorite', 'comment', 'follow']  # 复赛

# 各阶段构建数据集时的最后一天
# 15天代表测试集
STAGE_END_DAY = {"online_train": 14, "offline_train": 13, "evaluate": 14, "submit": 15}

# 各行为构建训练数据时考虑的天数
ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5}  # 初赛
# ACTION_DAY_NUM = {"read_comment": 5, "like": 5, "click_avatar": 5, "forward": 5,
#                   "comment": 5, "follow": 5, "favorite": 5}  # 复赛

# 负样本的采样率(下采样后负样本数/原负样本数)
ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1}  # 初赛
# ACTION_SAMPLE_RATE = {"read_comment": 0.2, "like": 0.2, "click_avatar": 0.2, "forward": 0.1,
#                       "comment": 0.1, "follow": 0.1, "favorite": 0.1}  # 复赛

# 随机种子
SEED = 2021

# 测试集
END_DAY = 15

# 特征列
FEATURE_COLUMNS = [
    # context
    'userid', 'feedid', 'device',

    # user
    'login_day_count', 'avg_play_user', 'avg_stay_user', 'avg_interact_user', 'avg_videoplayseconds',
    'browse_count', 'read_comment_count_user', 'read_comment_rate_user', 'like_count_user',
    'like_rate_user', 'click_avatar_count_user', 'click_avatar_rate_user', 'forward_count_user',
    'forward_rate_user', 'finish_count_user', 'finish_rate_user',

    # feed
    'authorid', 'bgm_singer_id', 'bgm_song_id', 'videoplayseconds', 'avg_play_feed',
    'avg_stay_feed', 'avg_interact_feed', 'show_count', 'read_comment_count_feed',
    'read_comment_rate_feed', 'like_count_feed', 'like_rate_feed', 'click_avatar_count_feed',
    'click_avatar_rate_feed', 'forward_count_feed', 'forward_rate_feed', 'finish_count_feed',
    'finish_rate_feed',
]
SPARSE_FEATURE_COLUMNS = ['userid', 'feedid', 'device', 'authorid', 'bgm_singer_id', 'bgm_song_id']



