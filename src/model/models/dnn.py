# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-06-02
# @Contact : liaozhi_edo@163.com


"""
    The Multilayer Perceptron
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from src.model.layers.utils import concat_func
from src.model.layers.core import DNN as BASE_DNN
from src.model.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from src.model.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def DNN(feature_columns, hidden_units=(256, 128, 64), activation='relu', dropout_rate=0.5, use_bn=True):
    """
    DNN模型

    :param feature_columns: list 特征列
    :param hidden_units: tuple 神经元数目
    :param activation: str 激活函数
    :param dropout_rate: float dropout rate
    :param use_bn: Bool batch normalization
    :return:
    """
    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    # dense
    dense_value_list = get_dense_value(input_dict, feature_columns)

    # sparse
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # varlen sparse
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_feature_columns)
    hist = []

    # concat
    dnn_embedding_input = concat_func(sparse_embedding_list + seq_pooling_embedding_list + hist, mask=False)
    dnn_embedding_input = layers.Flatten()(dnn_embedding_input)
    dnn_input = concat_func(dense_value_list + [dnn_embedding_input], mask=False)

    # 4,DNN
    dnn = BASE_DNN(hidden_units=hidden_units, activation=activation, dropout_rate=dropout_rate, use_bn=use_bn)
    dnn_output = dnn(dnn_input)
    final_output = layers.Dense(1, activation='sigmoid', name='prediction')(dnn_output)

    # 5,model
    model = tf.keras.Model(
        inputs=list(input_dict.values()),
        outputs=final_output
    )

    return model
