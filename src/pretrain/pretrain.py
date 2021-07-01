# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-07-01
# @Contact : liaozhi_edo@163.com


"""
    预训练Embedding以及部分离散ID类特征的处理(获取mapping)
"""

# packages
import numpy as np
import pandas as pd
from functools import reduce
from gensim.models import Word2Vec
from config.conf import *


def get_feed_embedding():
    """
    feed embedding特征

    :return:
    """
    # 1,加载数据
    feed_embed = pd.read_csv(DATA_HOME + 'feed_embeddings.csv', header=0, index_col=False)
    feed_embed['feed_embedding'] = feed_embed['feed_embedding'].apply(lambda x: [eval(_) for _ in x.strip().split(' ')])

    # 2,数据处理
    feed_map = dict()
    feed_list = feed_embed['feedid'].unique().tolist()
    feed_embedding_matrix = np.random.uniform(size=(len(feed_list) + 1, 512))  # matrix[0] for NAN
    for idx, feed in enumerate(feed_list, 1):
        feed_map[feed] = idx
        feed_embedding_matrix[idx] = np.array(feed_embed.loc[feed_embed.feedid == feed, 'feed_embedding'].tolist()[0])

    # 3,保存
    file_path = join(SAVE_HOME, 'feature', 'feed_embedding.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(feed_map, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feed_embedding_matrix, file, pickle.HIGHEST_PROTOCOL)

    return feed_embedding_matrix, feed_map


def process_sequence_feature(id_list, id_map=None, maxlen=5):
    """
    处理序列特征

    :param id_list: Series id序列
    :param id_map: dict 类别编码
    :param maxlen: int 序列长度
    :return:
    """
    # 1,数据类型判断
    if not isinstance(id_list, list):
        return [0] * maxlen  # 0 is a mask value

    # 2,类别编码
    if id_map:
        idx_list = [id_map.get(id_, 0) for id_ in id_list]
    else:
        idx_list = id_list

    # 3,padding
    if len(idx_list) >= maxlen:
        idx_list = idx_list[:maxlen]
    else:
        idx_list = np.pad(idx_list, pad_width=(0, maxlen - len(idx_list)), constant_values=0).tolist()

    return idx_list


def process_feed_tag():
    """
    处理feed的类别标签

    :return:
    """
    # 1,加载数据
    columns = ['feedid', 'manual_tag_list', 'machine_tag_list']
    feed_tag = pd.read_csv(DATA_HOME + 'feed_info.csv', header=0, index_col=False, usecols=columns)
    feed_tag.rename(columns={'machine_tag_list': 'machine_tag_weight_pair_list'}, inplace=True)

    # 2,数据处理
    feed_tag['manual_tag_list'] = feed_tag['manual_tag_list'].str.split(';')
    feed_tag['manual_tag_list'] = feed_tag['manual_tag_list'].apply(lambda x: x if isinstance(x, list) else [])
    feed_tag['machine_tag_weight_pair_list'] = feed_tag['machine_tag_weight_pair_list'].str.split(';')
    feed_tag['machine_tag_list'] = feed_tag['machine_tag_weight_pair_list'].apply(
        lambda x: [tag_weight_pair.split(' ')[0] for tag_weight_pair in x] if isinstance(x, list) else [])
    feed_tag['tag_list'] = feed_tag[['manual_tag_list', 'machine_tag_list']].apply(
        lambda S: list(set(S['manual_tag_list']) | set(S['machine_tag_list'])), axis=1)
    feed_tag['machine_tag_list_weight'] = feed_tag['machine_tag_weight_pair_list'].apply(
        lambda x: [eval(tag_weight_pair.split(' ')[1]) for tag_weight_pair in x] if isinstance(x, list) else [])
    feed_tag.drop(columns=['machine_tag_weight_pair_list'], inplace=True)

    # tag map
    tag_list = feed_tag['manual_tag_list'].tolist() + feed_tag['machine_tag_list'].tolist()
    tag_list = reduce(lambda x, y: set(x) | set(y), tag_list)
    tag_map = dict()
    for idx, tag in enumerate(tag_list, 1):
        tag_map[tag] = idx

    # 3,构建序列特征
    feed_tag['manual_tag_list'] = feed_tag['manual_tag_list'].apply(
        func=process_sequence_feature,
        id_map=tag_map,
        maxlen=MAXLEN['manual_tag_list']
    )
    feed_tag['machine_tag_list'] = feed_tag['machine_tag_list'].apply(
        func=process_sequence_feature,
        id_map=tag_map,
        maxlen=MAXLEN['machine_tag_list']
    )
    feed_tag['machine_tag_list_weight'] = feed_tag['machine_tag_list_weight'].apply(
        func=process_sequence_feature,
        id_map=None,
        maxlen=MAXLEN['machine_tag_list']
    )
    feed_tag['tag_list'] = feed_tag['tag_list'].apply(
        func=process_sequence_feature,
        id_map=tag_map,
        maxlen=MAXLEN['tag_list']
    )

    # 4,保存
    file_path = join(SAVE_HOME, 'feature', 'feed_tag.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(feed_tag, file, pickle.HIGHEST_PROTOCOL)

    return feed_tag, tag_map


def process_feed_keyword():
    """
    处理feed的keyword特征

    :return:
    """
    # 1,加载数据
    columns = ['feedid', 'manual_keyword_list', 'machine_keyword_list']
    feed_keyword = pd.read_csv(DATA_HOME + 'feed_info.csv', header=0, index_col=False, usecols=columns)

    # 2,数据处理
    feed_keyword['manual_keyword_list'] = feed_keyword['manual_keyword_list'].str.split(';')
    feed_keyword['manual_keyword_list'] = feed_keyword['manual_keyword_list'].apply(
        lambda x: x if isinstance(x, list) else [])
    feed_keyword['machine_keyword_list'] = feed_keyword['machine_keyword_list'].str.split(';')
    feed_keyword['machine_keyword_list'] = feed_keyword['machine_keyword_list'].apply(
        lambda x: x if isinstance(x, list) else [])
    feed_keyword['keyword_list'] = feed_keyword[['manual_keyword_list', 'machine_keyword_list']].apply(
        lambda S: list(set(S['manual_keyword_list']) | set(S['machine_keyword_list'])), axis=1)

    # keyword map
    keyword_set = set()
    for _, row in feed_keyword[['manual_keyword_list', 'machine_keyword_list']].iterrows():
        keyword_set |= set(row['manual_keyword_list']) | set(row['machine_keyword_list'])
    keyword_map = dict()
    for idx, keyword in enumerate(keyword_set, 1):
        keyword_map[keyword] = idx

    # 3,构建序列特征
    feed_keyword['manual_keyword_list'] = feed_keyword['manual_keyword_list'].apply(
        func=process_sequence_feature,
        id_map=keyword_map,
        maxlen=MAXLEN['manual_keyword_list']
    )
    feed_keyword['machine_keyword_list'] = feed_keyword['machine_keyword_list'].apply(
        func=process_sequence_feature,
        id_map=keyword_map,
        maxlen=MAXLEN['machine_keyword_list']
    )
    feed_keyword['keyword_list'] = feed_keyword['keyword_list'].apply(
        func=process_sequence_feature,
        id_map=keyword_map,
        maxlen=MAXLEN['keyword_list']
    )

    # 4,保存
    file_path = join(SAVE_HOME, 'feature', 'feed_keyword.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(feed_keyword, file, pickle.HIGHEST_PROTOCOL)

    return feed_keyword, keyword_map


def get_feedid_embedding_w2v():
    """
    利用Word2Vector获取feedid的embedding

    注意:
        1,冷启动feedid其label encode为0, embedding初始化为uniform.

    :return:
    """
    # 1,加载数据并构建序列
    # 注意:此处的id由int变为str
    columns = ['userid', 'feedid']
    user_action = pd.read_csv(DATA_HOME + 'user_action.csv', header=0, index_col=False, usecols=columns)
    test = pd.read_csv(DATA_HOME + 'test_b.csv', header=0, index_col=False, usecols=columns)
    user_action = pd.concat([user_action, test], ignore_index=False, sort=False)
    sentences = user_action.groupby(by=['userid'])['feedid'].apply(lambda x: [str(_) for _ in x.tolist()]).tolist()

    # 2,Word2Vector模型训练
    model = Word2Vec(sentences=sentences, size=EMBEDDING_DIM,
                     window=8, min_count=1, workers=8, sg=1, hs=0,
                     negative=20, ns_exponent=0.75, iter=10)

    # 3,获取embedding
    feed_info = pd.read_csv(DATA_HOME + 'feed_info.csv', header=0, index_col=False, usecols=['feedid'])
    feedid_list = feed_info['feedid'].unique().tolist()
    feedid_map = dict()
    feedid_embedding_matrix = np.random.uniform(size=(len(feedid_list) + 1, EMBEDDING_DIM))  # matrix[0] for NAN
    for idx, feedid in enumerate(feedid_list, 1):
        feedid_map[feedid] = idx
        if str(feedid) in model.wv.index2word:
            feedid_embedding_matrix[idx] = model.wv[str(feedid)]

    # 4,保存
    file_path = join(SAVE_HOME, 'feature', 'feedid_embedding_w2v.pkl')
    with open(file_path, 'wb') as file:
        pickle.dump(feedid_map, file, pickle.HIGHEST_PROTOCOL)
        pickle.dump(feedid_embedding_matrix, file, pickle.HIGHEST_PROTOCOL)

    return feedid_embedding_matrix, feedid_map


def pretrain():
    """
    预训练Embedding以及部分离散ID类特征的处理(获取mapping)

    :return:
    """
    # 1,feed embedding(图+文等多媒体)
    feed_embedding_matrix, feed_map = get_feed_embedding()

    # 2,feed tag
    feed_tag, tag_map = process_feed_tag()

    # 3,feed keyword
    feed_keyword, keyword_map = process_feed_keyword()

    # 4,feedid embedding(W2V)
    feedid_embedding_matrix, feedid_map = get_feedid_embedding_w2v()

    return
