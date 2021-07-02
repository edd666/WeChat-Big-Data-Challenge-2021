# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-04-22
# @Contact : liaozhi_edo@163.com


"""
    Gated network
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.activation import activation_layer


class ExpertUtilizationDropoutLayer(layers.Layer):
    """
    Helper layer for MMOE layer, allows to drop some of the experts and then normalize other experts.

    Input shape
        - A tensor with shape: (batch_size, num_experts).

    Output shape
        - A tensor with shape: (batch_size, num_experts).

    References
        - Recommending What Video to Watch Next: A Multitask Ranking System[C].
    """
    def __init__(self, dropout_rate, **kwargs):
        super(ExpertUtilizationDropoutLayer, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate
        # inside params
        self.dropout_layer = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(ExpertUtilizationDropoutLayer, self).build(input_shape)

        # dropout layer
        self.dropout_layer = layers.Dropout(self.dropout_rate, noise_shape=(1, *input_shape[1:]))

    def call(self, inputs, training=None, **kwargs):
        # dropout
        expert_prob_drop = self.dropout_layer(inputs, training)

        # normalizer
        normalizer = tf.add(tf.reduce_sum(expert_prob_drop, axis=-1, keepdims=True),
                            tf.keras.backend.epsilon())

        return expert_prob_drop / normalizer

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'dropout_rate': self.dropout_rate
        }
        base_config = super(ExpertUtilizationDropoutLayer, self).get_config()
        base_config.update(config)

        return base_config


class MultiGateMixtureOfExpertsLayer(layers.Layer):
    """
    The Multi-Gate Mixture of Experts(MMOE), which explicitly learns to model task relationships from data.
    It adapt the Mixture-of-Experts(MoE) structure to multi-task learning by sharing the expert sub-models
    across all tasks, while also having a gating network trained to optimize each task.

    Input shape
        - A tensor with shape: (batch_size, input_dim).

    Output shape
        - A tensor with shape: (batch_size, expert_output_dim).

    References
        - Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts[C].
        - Recommending What Video to Watch Next: A Multitask Ranking System[C].
    """
    def __init__(self, experts, num_tasks, add_dropout=None, dropout_rate=0.1, **kwargs):
        """

        :param experts: list of Layer/Model 专家层
        :param num_tasks: int 子任务数量
        :param add_dropout: bool 是否dropout
        :param dropout_rate: float dropout rate
        :param kwargs:
        """
        super(MultiGateMixtureOfExpertsLayer, self).__init__(**kwargs)
        self.experts = experts
        self.num_experts = len(experts)
        self.num_tasks = num_tasks
        self.add_dropout = add_dropout
        self.dropout_rate = dropout_rate
        if self.add_dropout:
            self.drop_expert_layer = ExpertUtilizationDropoutLayer(self.dropout_rate)
        # inside params
        self.kernels = None
        self.activation_layers = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(MultiGateMixtureOfExpertsLayer, self).build(input_shape)

        # gating network for each task
        self.kernels = [
            self.add_weight(
                name='kernel' + str(i),
                shape=(input_shape[-1], self.num_experts),
                initializer='random_normal',
                trainable=True,
            )
            for i in range(self.num_tasks)
        ]
        self.activation_layers = [activation_layer('softmax') for _ in range(self.num_tasks)]

    def call(self, inputs, training=None, **kwargs):

        # compute each expert output
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(inputs, training=training))

        # compute each task input
        # 同一个样本对不同专家层的利用率(权重)的和为1
        task_inputs = []
        for i in range(self.num_tasks):
            # compute the probability of expert based on given input
            expert_utilization_prob = tf.tensordot(inputs, self.kernels[i], axes=(-1, 0))
            expert_utilization_prob = self.activation_layers[i](expert_utilization_prob)

            # dropout
            if self.add_dropout:
                expert_utilization_prob = self.drop_expert_layer(expert_utilization_prob, training=training)

            task_input = 0
            for j, expert_output in enumerate(expert_outputs):
                task_input += expert_output * tf.expand_dims(expert_utilization_prob[:, j], axis=-1)

            task_inputs.append(task_input)

        return task_inputs

    def get_config(self):
        config = {
            'experts': [layers.serialize(expert) for expert in self.experts],
            'num_experts': self.num_experts,
            'num_tasks': self.num_tasks,
            'add_dropout': self.add_dropout,
            'dropout_rate': self.dropout_rate,
            'drop_expert_layer': layers.serialize(self.drop_expert_layer) if self.add_dropout else None,
        }
        base_config = super(MultiGateMixtureOfExpertsLayer, self).get_config()
        base_config.update(config)

        return base_config
