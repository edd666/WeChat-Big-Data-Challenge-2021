# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-04-22
# @Contact : liaozhi_edo@163.com


"""
    《Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts》
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.core import DNN
from deeprec.layers.utils import concat_func
from deeprec.layers.gate import MultiGateMixtureOfExpertsLayer
from deeprec.layers.sequence import AttentionSequencePoolingLayer
from deeprec.feature_column import SparseFeat, VarLenSparseFeat, build_input_dict
from deeprec.inputs import build_embedding_dict, get_dense_value, embedding_lookup, get_varlen_pooling_list


def MMOE(feature_columns, behavior_columns, task_models, att_hidden_units=(64, 16),
         att_activation='Dice', att_weight_normalization=False, num_experts=4,
         expert_hidden_units=(64,), add_dropout=False, dropout_rate=0.1):
    """
    MMOE模型

    注意:
        1,feature_columns中特征的相对顺序关系,如item_id,cate_id,其对应的行为序列为
            hist_item_id,hist_item_id.(主要是attention的时候特征要对齐)

    :param feature_columns: list 特征列
    :param behavior_columns: list 行为特征名称,表示哪些特征需要进行Attention
    :param task_models: dict 任务网络, 形如{task_name: Model}
    :param att_hidden_units: tuple Attention中DNN神经元数
    :param att_activation: str Attention中DNN的激活函数
    :param att_weight_normalization: bool Attention中score是否归一化
    :param num_experts: int 专家个数
    :param expert_hidden_units: int 专家隐层神经元数量
    :param add_dropout: bool gating network是否dropout
    :param dropout_rate: float gating network的dropout rate
    :return:
    """
    # 1,构建输入字典
    input_dict = build_input_dict(feature_columns)

    # 2,构建Embedding字典
    embedding_dict = build_embedding_dict(feature_columns, seq_mask_zero=True)

    # 3,构建模型的输入
    # dense
    dense_value_list = get_dense_value(input_dict, feature_columns)

    # sparse + seq(varlen sparse)
    sparse_feature_columns = list(filter(lambda x: isinstance(x, SparseFeat), feature_columns))
    seq_feature_columns = list(filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns))
    if len(behavior_columns) == 0:
        # 序列特征全部做Pooling
        sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)
        seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_feature_columns)
        hist = []
    else:
        # 序列特征做Pooling或Attention
        hist_behavior_columns = ['hist_' + str(col) for col in behavior_columns]
        query_feature_columns = [fc for fc in sparse_feature_columns if fc.name in behavior_columns]
        keys_feature_columns = [fc for fc in seq_feature_columns if fc.name in hist_behavior_columns]
        assert len(behavior_columns) == len(query_feature_columns) == len(keys_feature_columns)

        # sparse
        sparse_feature_columns = [fc for fc in sparse_feature_columns if fc.name not in behavior_columns]
        sparse_embedding_list = embedding_lookup(input_dict, embedding_dict, sparse_feature_columns, to_list=True)

        # seq pooling
        seq_pooling_feature_columns = [fc for fc in seq_feature_columns if fc.name not in hist_behavior_columns]
        seq_pooling_embedding_list = get_varlen_pooling_list(input_dict, embedding_dict, seq_pooling_feature_columns)

        # seq attention
        query_embedding_list = embedding_lookup(input_dict, embedding_dict, query_feature_columns, to_list=True)
        query = concat_func(query_embedding_list, mask=True)
        keys_embedding_list = embedding_lookup(input_dict, embedding_dict, keys_feature_columns, to_list=True)
        keys = concat_func(keys_embedding_list, mask=True)
        hist = AttentionSequencePoolingLayer(
            hidden_units=att_hidden_units,
            activation=att_activation,
            mask_zero=True,
            weight_normalization=att_weight_normalization,
            return_score=False)([query, keys])
        hist = [hist]

    # concat
    dnn_embedding_input = concat_func(sparse_embedding_list + seq_pooling_embedding_list + hist, mask=False)
    dnn_embedding_input = layers.Flatten()(dnn_embedding_input)
    dnn_input = concat_func(dense_value_list + [dnn_embedding_input], mask=False)

    # 4,Multi-gate Mixture-of-Experts
    experts = [
        DNN(
            hidden_units=expert_hidden_units,
            activation='relu',
            use_bn=False,
            dropout_rate=0,
        )
        for _ in range(num_experts)
    ]
    task_inputs = MultiGateMixtureOfExpertsLayer(
        experts=experts,
        num_tasks=len(task_models),
        add_dropout=add_dropout,
        dropout_rate=dropout_rate,)(dnn_input)

    # 5,子任务的输出
    task_outputs = list()
    for idx, (name, task_model) in enumerate(task_models.items()):
        task_outputs.append(task_model(task_inputs[idx]))

    # 6,functional model
    model = tf.keras.Model(
        inputs=input_dict,
        outputs=task_outputs,
    )

    return model
