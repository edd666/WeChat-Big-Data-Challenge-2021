# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    《Deep Interest Network for Click-Through Rate Prediction》
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.core import DNN
from deeprec.layers.utils import concat_func
from deeprec.layers.sequence import AttentionSequencePoolingLayer
from deeprec.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def DIN(feature_columns, behavior_columns, att_hidden_units=(64, 16), att_activation='Dice',
        att_weight_normalization=False, dnn_hidden_units=(256, 128), dnn_activation='relu',
        dnn_dropout_rate=0.5, dnn_use_bn=True):
    """
    DIN模型

    注意:
        1,feature_columns中特征的相对顺序关系,如item_id,cate_id,其对应的行为序列为
            hist_item_id,hist_item_id.(主要是attention的时候特征要对齐)

    :param feature_columns: list 特征列
    :param behavior_columns: list 行为序列(Attention)的特征名称
    :param att_hidden_units: tuple Attention中DNN神经元数
    :param att_activation: str Attention中DNN的激活函数
    :param att_weight_normalization: bool Attention中score是否归一化
    :param dnn_hidden_units: tuple DNN模型各层神经元数量
    :param dnn_activation: str DNN模型激活函数
    :param dnn_dropout_rate: float DNN模型dropout_rate
    :param dnn_use_bn: bool DNN模型是否使用BatchNormalization
    :return:
    """
    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    dense_value_list = get_dense_value(input_dict, feature_columns)

    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_feature_columns = [fc for fc in sparse_feature_columns if fc.name not in behavior_columns]
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # seq = varlen sparse
    # pooling
    hist_behavior_columns = ['hist_' + str(col) for col in behavior_columns]
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    seq_pooling_feature_columns = [fc for fc in seq_feature_columns if fc.name not in hist_behavior_columns]
    seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_pooling_feature_columns)

    # attention
    query_feature_columns = [fc for fc in feature_columns if fc.name in behavior_columns]
    query_embedding_list = embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=True)
    query = concat_func(query_embedding_list, mask=True)
    keys_feature_columns = [fc for fc in feature_columns if fc.name in hist_behavior_columns]
    keys_embedding_list = embedding_lookup(input_dict, embedding_dict, keys_feature_columns, to_list=True)
    keys = concat_func(keys_embedding_list, mask=True)
    hist = AttentionSequencePoolingLayer(
        hidden_units=att_hidden_units,
        activation=att_activation,
        mask_zero=True,
        weight_normalization=att_weight_normalization,
        return_score=False)([query, keys])

    # concat
    dnn_embedding_input = concat_func(sparse_embedding_list + seq_pooling_embedding_list + [hist], mask=False)
    dnn_embedding_input = layers.Flatten()(dnn_embedding_input)
    dnn_input = concat_func(dense_value_list + [dnn_embedding_input], mask=False)

    # 4,DNN
    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        dropout_rate=dnn_dropout_rate,
        use_bn=dnn_use_bn,)(dnn_input)
    dnn_output = layers.Dense(1, activation='sigmoid', name='ctr_output')(dnn_output)

    # 5,model
    model = tf.keras.Model(
        inputs=input_dict,
        outputs=dnn_output
    )

    return model


