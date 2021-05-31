# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-04-06
# @Contact : liaozhi_edo@163.com


"""
    《Deep Interest Evolution Network for Click-Through Rate Prediction》
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.core import DNN
from deeprec.layers.utils import concat_func
from deeprec.layers.cells import AGRUCell, AUGRUCell
from deeprec.layers.sequence import AttentionSequencePoolingLayer
from deeprec.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def auxiliary_loss(state_seq, click_seq, noclick_seq, mask):
    """
    兴趣提取层(interest extraction)的辅助loss

    :param state_seq: tensor (B, T-1, units) 兴趣提取层(GRU)的输出
    :param click_seq: tensor (B, T-1, E) 点击序列
    :param noclick_seq: tensor (B, T-1, E) 随机采样获得非点击序列
    :param mask: tensor (B, T-1) 点击序列mask
    :return:
    """
    # 1,auxiliary net
    auxiliary_net = tf.keras.Sequential(
        [
            layers.BatchNormalization(),
            layers.Dense(100, activation='sigmoid'),
            layers.Dense(50, activation='sigmoid'),
            layers.Dense(1, activation='sigmoid')
        ],
        name='auxiliary_net'
    )

    # 2,prob(click)
    click_input = concat_func([state_seq, click_seq], mask=False)
    noclick_input = concat_func([state_seq, noclick_seq], mask=False)
    click_prob = auxiliary_net(click_input)[:, :, 0]  # (batch_size, seq_len)
    noclick_prob = auxiliary_net(noclick_input)[:, :, 0]

    # 3,aux_loss
    mask = tf.cast(mask, 'float32')
    click_loss = tf.reshape(tf.math.log(click_prob), [-1, click_seq.get_shape()[1]]) * mask
    noclick_loss = tf.reshape(tf.math.log(1.0 - noclick_prob), [-1, noclick_seq.get_shape()[1]]) * mask
    aux_loss = tf.math.reduce_mean(tf.math.reduce_sum(click_loss + noclick_loss, axis=-1))

    return aux_loss


def interest_evolution(behavior_concat, item_dnn_input, neg_behavior_concat=None,
                       use_neg=True, gru_type='AUGRU', att_hidden_units=(64, 16),
                       att_activation='Dice', att_weight_normalization=False, alpha=1.0):
    """
    兴趣演化层(interest evolution),包含兴趣提取层(interest extraction)

    :param behavior_concat: tensor (B, T, E) 点击序列(keys)
    :param item_dnn_input: tensor (B, 1, E) 物品向量(query)
    :param neg_behavior_concat: tensor (B, T, E) 随机采样得到非点击序列(neg_keys)
    :param use_neg: bool 是否使用负样本来训练兴趣提取层
    :param gru_type: str GRU
    :param att_hidden_units: tuple Attention中神经元个数
    :param att_activation: str Attention的激活函数
    :param att_weight_normalization: bool Attention中score是否归一化
    :param alpha: float auxiliary loss(interest extraction layer)的权重
    :return:
    """
    # 1,params
    if gru_type not in ('GRU', 'AIGRU', 'AGRU', 'AUGRU'):
        raise ValueError('Invalid gru_type.')
    # the dim of hidden state of GRU
    units = behavior_concat.get_shape()[-1]

    # 2,interest extraction
    # RNN support masking
    interest_extraction_rnn = layers.RNN(layers.GRUCell(units), return_sequences=True)
    # the sequence of hidden state of the interest extraction layer
    rnn_outputs_1 = interest_extraction_rnn(behavior_concat)

    # auxiliary loss
    if use_neg:
        if neg_behavior_concat is None:
            raise ValueError('when use_neg is True, neg_behavior_concat should be a tensor.')
        aux_loss = auxiliary_loss(
            state_seq=rnn_outputs_1[:, : -1, :],
            click_seq=behavior_concat[:, 1:, :],
            noclick_seq=neg_behavior_concat[:, 1:, :],
            mask=behavior_concat._keras_mask[:, 1:],
        )
        interest_extraction_rnn.add_loss(alpha * aux_loss)

    # 3,interest evolution
    # hist is the final output of rnn and the final state of rnn
    if gru_type == 'GRU':
        # GRU
        interest_evolution_rnn = layers.RNN(layers.GRUCell(units), return_sequences=True)
        rnn_outputs_2 = interest_evolution_rnn(rnn_outputs_1)

        # Attention
        hist = AttentionSequencePoolingLayer(
            hidden_units=att_hidden_units,
            activation=att_activation,
            mask_zero=True,
            weight_normalization=att_weight_normalization,
            return_score=False)([item_dnn_input, rnn_outputs_2])
    else:
        # AIGRU, AGRU, AUGRU
        att_score = AttentionSequencePoolingLayer(
            hidden_units=att_hidden_units,
            activation=att_activation,
            mask_zero=True,
            weight_normalization=att_weight_normalization,
            return_score=True)([item_dnn_input, rnn_outputs_1])

        if gru_type == 'AIGRU':
            # AIGRU
            interest_evolution_rnn = layers.RNN(layers.GRUCell(units))
            hist = interest_evolution_rnn(rnn_outputs_1 * layers.Permute([2, 1])(att_score))
        else:
            # AGRU, AUGRU
            cell = AGRUCell(units) if gru_type == 'AGRU' else AUGRUCell(units)
            interest_evolution_rnn = layers.RNN(cell)
            hist = interest_evolution_rnn((rnn_outputs_1, layers.Permute([2, 1])(att_score)))

        # append a dimension
        hist = tf.expand_dims(hist, axis=1)

    return hist


def DIEN(feature_columns, behavior_columns, use_neg=True, gru_type='AUGRU',
         att_hidden_units=(64, 16), att_activation='Dice', att_weight_normalization=False,
         dnn_hidden_units=(256, 128), dnn_activation='relu', dnn_dropout_rate=0.5, dnn_use_bn=True,
         alpha=1.0):
    """
    DIEN模型

    注意:
        1,feature_columns中特征的相对顺序关系,如item_id,cate_id,其对应的行为序列为
            hist_item_id,hist_item_id.(主要是attention的时候特征要对齐)

    :param feature_columns: list 特征列
    :param behavior_columns: list list 行为序列(Attention)的特征名称
    :param use_neg: bool 是否采用负样本训练兴趣提取层
    :param gru_type: str GRU
    :param att_hidden_units: tuple Attention中神经元个数
    :param att_activation: str Attention的激活函数
    :param att_weight_normalization: bool Attention中score是否归一化
    :param dnn_hidden_units: tuple DNN模型各层神经元数量
    :param dnn_activation: str DNN模型激活函数
    :param dnn_dropout_rate: float DNN模型dropout_rate
    :param dnn_use_bn: bool DNN模型是否使用BatchNormalization
    :param alpha: float auxiliary loss(interest extraction layer)的权重
    :return:
    """
    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型输入
    # dense
    dense_value_list = get_dense_value(input_dict, feature_columns)

    # sparse
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    sparse_feature_columns = [fc for fc in sparse_feature_columns if fc.name not in behavior_columns]
    sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

    # seq = varlen sparse
    # pooling
    hist_feature_columns = []
    neg_hist_feature_columns = []
    seq_pooling_feature_columns = []
    hist_behavior_columns = ['hist_' + str(col) for col in behavior_columns]
    neg_hist_behavior_columns = ['neg_' + str(col) for col in hist_behavior_columns]
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    for fc in seq_feature_columns:
        if fc.name in hist_behavior_columns:
            hist_feature_columns.append(fc)
        elif fc.name in neg_hist_behavior_columns:
            neg_hist_feature_columns.append(fc)
        else:
            seq_pooling_feature_columns.append(fc)

    seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_pooling_feature_columns)

    # GRU + Attention
    query_feature_columns = [fc for fc in feature_columns if fc.name in behavior_columns]
    query_embedding_list = embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=True)
    query = concat_func(query_embedding_list, mask=True)
    keys_embedding_list = embedding_lookup(input_dict, embedding_dict, hist_feature_columns, to_list=True)
    keys = concat_func(keys_embedding_list, mask=True)
    if use_neg:
        # 利用随机采样构建负样本辅助interest extraction的学习
        if len(neg_hist_feature_columns) != 0:
            neg_keys_embedding_list = embedding_lookup(input_dict, embedding_dict, neg_hist_feature_columns,
                                                       to_list=True)
            neg_keys = concat_func(neg_keys_embedding_list, mask=True)
        else:
            raise ValueError('when use_neg is True, should include neg behavior feat in feature_columns.')
    else:
        neg_keys = None
    hist = interest_evolution(
        behavior_concat=keys,
        item_dnn_input=query,
        neg_behavior_concat=neg_keys,
        use_neg=use_neg,
        gru_type=gru_type,
        att_hidden_units=att_hidden_units,
        att_activation=att_activation,
        att_weight_normalization=att_weight_normalization,
        alpha=alpha,
    )

    # concat
    dnn_embedding_input = concat_func(sparse_embedding_list + seq_pooling_embedding_list + [hist], mask=False)
    dnn_embedding_input = layers.Flatten()(dnn_embedding_input)
    dnn_input = concat_func(dense_value_list + [dnn_embedding_input], mask=False)

    # 4,DNN
    dnn_output = DNN(
        hidden_units=dnn_hidden_units,
        activation=dnn_activation,
        dropout_rate=dnn_dropout_rate,
        use_bn=dnn_use_bn, )(dnn_input)
    dnn_output = layers.Dense(1, activation='sigmoid', name='ctr_output')(dnn_output)

    # 5,model
    model = tf.keras.Model(
        inputs=input_dict,
        outputs=dnn_output
    )

    return model
