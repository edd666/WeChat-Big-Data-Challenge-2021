# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-11
# @Contact : liaozhi_edo@163.com


"""
    序列处理相关的layer
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.core import LocalActivationUnit


class SequencePoolingLayer(layers.Layer):
    """
    The SequencePoolingLayer is used to apply pooling operation(sum,mean,max) on variable-length
    sequence feature/multi-value feature.

    Input shape
        - A list of two tensor [seq_value, seq_len]
        - seq_value is a 3D tensor with shape: (batch_size, T, embedding_size)
        - seq_len is a 2D tensor with shape : (batch_size, 1),indicate valid length of each sequence.

    Output shape
        - 3D tensor with shape: (batch_size, 1, embedding_size).
    """
    def __init__(self, mode, mask_zero=True, **kwargs):
        """

        :param mode: str Pooling方法
        :param mask_zero: bool 是否支持mask zero
        :param kwargs:
        :return:
        """
        super(SequencePoolingLayer, self).__init__(**kwargs)
        if mode not in ('sum', 'mean', 'max'):
            raise ValueError("mode must be sum, mean or max")
        self.mode = mode
        self.mask_zero = mask_zero
        self.seq_maxlen = None
        self.eps = tf.constant(1e-8, tf.float32)

    def build(self, input_shape):
        if not self.mask_zero:
            self.seq_maxlen = int(input_shape[0][1])
        super(SequencePoolingLayer, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        if self.mask_zero:
            if mask is None:
                raise ValueError("When mask_zero=True,input must support masking.")
            seq_value = inputs
            mask = tf.cast(mask, tf.float32)  # tf.to_float(mask)
            seq_len = tf.math.reduce_sum(mask, axis=-1, keepdims=True)  # (batch_size, 1)
            mask = tf.expand_dims(mask, axis=2)
        else:
            seq_value, seq_len = inputs
            mask = tf.sequence_mask(seq_len, self.seq_maxlen, dtype=tf.float32)
            mask = tf.transpose(mask, (0, 2, 1))

        embedding_size = seq_value.shape[-1]
        mask = tf.tile(mask, [1, 1, embedding_size])

        # max
        if self.mode == 'max':
            seq_value = seq_value - (1 - mask) * 1e9
            return tf.math.reduce_max(seq_value, 1, keepdims=True)

        # sum
        seq_value = tf.math.reduce_sum(seq_value * mask, 1, keepdims=False)

        # mean
        if self.mode == 'mean':
            seq_value = tf.math.divide(seq_value, tf.cast(seq_len, dtype=tf.float32) + self.eps)

        seq_value = tf.expand_dims(seq_value, axis=1)

        return seq_value

    def compute_output_shape(self, input_shape):
        if self.mask_zero:
            embedding_size = input_shape[-1]
        else:
            embedding_size = input_shape[0][-1]

        shape = (None, 1, embedding_size)
        return shape

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {'mode': self.mode, 'mask_zero': self.mask_zero}
        base_config = super(SequencePoolingLayer, self).get_config()
        base_config.update(config)

        return base_config


class AttentionSequencePoolingLayer(layers.Layer):
    """
    The Attentional sequence pooling operation used in DIN.

    Input shape
        - A list of 3 tensor [query, keys, keys_length]
        - query is a 3D tensor with shape: (batch_size, 1, embedding_size)
        - keys is a 3D tensor with shape: (batch_size, T, embedding_size)
        - keys_length is a 2D tensor with shape : (batch_size, 1),indicate valid length of each sequence.

    Output shape
        - 3D tensor with shape: (batch_size, 1, embedding_size).

    References
        - Deep interest network for click-through rate prediction[C] (https://arxiv.org/pdf/1706.06978.pdf).
    """
    def __init__(self, hidden_units=(36,), activation='Dice', mask_zero=True,
                 weight_normalization=False, return_score=False, **kwargs):
        """

        :param hidden_units: list 各层神经元数量
        :param activation:  str 激活函数
        :param mask_zero: bool 是否支持mask
        :param weight_normalization: bool att_score归一化
        :param return_score: bool 是否返回att_score
        :param kwargs:
        """
        super(AttentionSequencePoolingLayer, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.mask_zero = mask_zero
        self.weight_normalization = weight_normalization
        self.return_score = return_score
        # inside params
        self.local_att = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(AttentionSequencePoolingLayer, self).build(input_shape)
        self.local_att = LocalActivationUnit(
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout_rate=0,
            use_bn=False)

    def call(self, inputs, mask=None, training=None, **kwargs):
        if self.mask_zero:
            if mask is None:
                raise ValueError("When mask_zero=True,input must support masking.")
            query, keys = inputs
            keys_mask = tf.expand_dims(mask[1], axis=1)
        else:
            query, keys, keys_length = inputs
            seq_len = keys.get_shape()[1]
            keys_mask = tf.sequence_mask(keys_length, seq_len)

        att_score = self.local_att([query, keys])
        att_score = tf.transpose(att_score, (0, 2, 1))

        if self.weight_normalization:
            padding = tf.ones_like(att_score) * (-2 ** 32 + 1)
        else:
            padding = tf.zeros_like(att_score)

        att_score = tf.where(keys_mask, att_score, padding)

        if self.weight_normalization:
            att_score = tf.nn.softmax(att_score)

        if not self.return_score:
            return tf.matmul(att_score, keys)

        return att_score

    def compute_output_shape(self, input_shape):
        if self.return_score:
            shape = (None, 1, input_shape[1][1])
        else:
            shape = (None, 1, input_shape[0][-1])

        return shape

    def compute_mask(self, inputs, mask=None):

        return None

    def get_config(self):
        config = {
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'return_score': self.return_score,
            'mask_zero': self.mask_zero,
            'weight_normalization': self.weight_normalization,
        }
        base_config = super(AttentionSequencePoolingLayer, self).get_config()
        base_config.update(config)

        return base_config
