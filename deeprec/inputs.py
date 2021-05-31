# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    模型输入层相关函数,如embedding_lookup
"""

# packages
from collections import OrderedDict
from tensorflow.keras import layers
from deeprec.layers.sequence import SequencePoolingLayer
from deeprec.feature_column import DenseFeat, SparseFeat, VarLenSparseFeat


def build_embedding_dict(feature_columns, seq_mask_zero=True):
    """
    基于特征列(feature columns)构建Embedding字典

    :param feature_columns: list 特征列
    :param seq_mask_zero: bool 序列输入是否mask zero
    :return:
        embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    """
    # 1,获取SparseFeat和VarLenSparseFeat
    sparse_feature_columns = list(filter(
        lambda x: isinstance(x, SparseFeat), feature_columns))
    varlen_sparse_feature_columns = list(filter(
        lambda x: isinstance(x, VarLenSparseFeat), feature_columns))

    # 2,构建Embedding字典
    embedding_dict = OrderedDict()
    for fc in sparse_feature_columns:
        embedding_dict[fc.embedding_name] = layers.Embedding(input_dim=fc.vocabulary_size,
                                                             output_dim=fc.embedding_dim,
                                                             embeddings_initializer=fc.embeddings_initializer,
                                                             trainable=fc.trainable,
                                                             name='sparse_emb_' + fc.embedding_name)

    for fc in varlen_sparse_feature_columns:
        embedding_dict[fc.embedding_name] = layers.Embedding(input_dim=fc.vocabulary_size,
                                                             output_dim=fc.embedding_dim,
                                                             embeddings_initializer=fc.embeddings_initializer,
                                                             trainable=fc.trainable,
                                                             mask_zero=seq_mask_zero,
                                                             name='varlen_sparse_emb_' + fc.embedding_name)
    return embedding_dict


def get_dense_value(input_dict, feature_columns):
    """
    获取数值输入

    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param feature_columns: list 特征列
    :return:
        dense_value_list: list 数值输入
    """
    # 1,获取DenseFeat
    dense_value_list = list()
    dense_feature_columns = list(filter(
        lambda x: isinstance(x, DenseFeat), feature_columns))
    for fc in dense_feature_columns:
        dense_value_list.append(input_dict[fc.name])

    return dense_value_list


def embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=False):
    """
    embedding查询

    注意:
        1,query_feature_columns可以是SparseFeat或VarLenSparseFeat
        2,input_dict和embedding_dict必须包含相应的输入和embedding table

    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param query_feature_columns: list 待查询的特征列
    :param to_list: bool 是否转成list
    :return:
    """
    # 1,查询
    query_embedding_dict = OrderedDict()
    for fc in query_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.use_hash:
            raise ValueError('hash embedding lookup has not yet been implemented.')
        else:
            lookup_idx = input_dict[feature_name]

        query_embedding_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)

    if to_list:
        return list(query_embedding_dict.values())

    return query_embedding_dict


def get_varlen_pooling_list(input_dict, embedding_dict, varlen_sparse_feature_columns):
    """
    对序列特征(VarLenSparseFeat)进行Pooling操作

    :param input_dict: dict 输入字典,形如{feature_name: keras.Input()}
    :param embedding_dict: embedding字典,形如{embedding_name: embedding_table}
    :param varlen_sparse_feature_columns: list 序列特征
    :return:
    """
    # 1,对VarLenSparseFeat的embedding进行Pooling操作
    pooling_value_list = []
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name
        if fc.weight_name is not None:
            raise ValueError('pooling with weight has not yet been implemented.')
        else:
            seq_value = embedding_dict[embedding_name](input_dict[feature_name])

        pooling_value = SequencePoolingLayer(mode=fc.combiner, mask_zero=True)(seq_value)
        pooling_value_list.append(pooling_value)

    return pooling_value_list

