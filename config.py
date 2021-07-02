# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    配置文件
"""

# packages
import os
import pickle
from os.path import join


# TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7,5'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Data
DATA_HOME = '/data_share/weixin/wechat_algo_data1/'
SAVE_HOME = '/data_share/weixin/lz/'


# Action
ACTION_LIST = ['read_comment', 'like', 'click_avatar', 'forward']  # 初赛


# Sample
STAGE_END_DAY = {'train': 14, 'test': 15}
ACTION_NUM_DAYS = {'read_comment': 14, 'like': 14, 'click_avatar': 14,
                   'forward': 14, 'favorite': 14, 'comment': 14, 'follow': 14}
ACTION_SAMPLE_RATE = {'read_comment': 1.0, 'like': 0.2, 'click_avatar': 0.2,
                      'forward': 0.1, 'favorite': 0.1, 'comment': 0.1, 'follow': 0.1}

# Feature
ONE_HOT_COLUMNS = ['device']  # 属于Dense feat
DENSE_FEATURE_COLUMNS = ['videoplayseconds']
# feed: feed embedding feedid: feedid embedding
SPARSE_FEATURE_COLUMNS = ['userid', 'feedid', 'authorid', 'bgm_song_id', 'bgm_singer_id', 'feed']
VARLEN_SPARSE_FEATURE_COLUMNS = ['manual_tag_list', 'machine_tag_list', 'keyword_list']

MAXLEN = {'manual_tag_list': 11, 'machine_tag_list': 14, 'tag_list': 14,
          'manual_keyword_list': 18, 'machine_keyword_list': 16, 'keyword_list': 22}
VOCABULARY_SIZE = {'manual_tag_list': 353, 'machine_tag_list': 353, 'tag_list': 353,
                   'manual_keyword_list': 27271, 'machine_keyword_list': 27271, 'keyword_list': 27271}
EMBEDDING_NAME = {'manual_tag_list': 'tag', 'machine_tag_list': 'tag', 'tag_list': 'tag',
                  'manual_keyword_list': 'keyword', 'machine_keyword_list': 'keyword', 'keyword_list': 'keyword'}
COMBINER = 'sum'
WEIGHT_NAME = {'machine_tag_list': 'machine_tag_list_weight'}


# Model
NUM_FLODS = 5
BATCH_SIZE = {'read_comment': 2048, 'like': 512, 'click_avatar': 512,
              'forward': 512, 'favorite': 512, 'comment': 512, 'follow': 512}
EMBEDDING_DIM = 64
ACTION_EPOCHS = {'read_comment': 10, 'like': 10, 'click_avatar': 10,
                 'forward': 10, 'favorite': 10, 'comment': 10, 'follow': 10}

# Evaluation
ACTION_WEIGHT = {'read_comment': 4, 'like': 3, 'click_avatar': 2,
                 'favorite': 1, 'forward': 1, 'comment': 1, 'follow': 1}


# Random Seed
SEED = 2021


# Load pre-processed data
# Tag
file_path = join(SAVE_HOME, 'feature', 'feed_tag.pkl')
with open(file_path, 'rb') as file:
    FEED_TAG = pickle.load(file)
del file_path, file

# Keyword
file_path = join(SAVE_HOME, 'feature', 'feed_keyword.pkl')
with open(file_path, 'rb') as file:
    FEED_KEYWORD = pickle.load(file)
del file_path, file

# Feed
file_path = join(SAVE_HOME, 'feature', 'feed_embedding.pkl')
with open(file_path, 'rb') as file:
    FEED_MAP = pickle.load(file)
    FEED_EMBEDDING_MATRIX = pickle.load(file)
    FEED_EMBEDDING_MATRIX = FEED_EMBEDDING_MATRIX.astype('float32')
del file_path, file

# Feedid
file_path = join(SAVE_HOME, 'feature', 'feedid_embedding_w2v.pkl')
with open(file_path, 'rb') as file:
    FEEDID_MAP = pickle.load(file)
    FEEDID_EMBEDDING_MATRIX = pickle.load(file)
    FEEDID_EMBEDDING_MATRIX = FEEDID_EMBEDDING_MATRIX.astype('float32')
del file_path, file
