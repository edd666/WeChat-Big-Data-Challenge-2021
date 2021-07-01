# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-15
# @Contact : liaozhi_edo@163.com


"""
    激活层
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers


class Dice(layers.Layer):
    """
    Data Adaptive Activation Function used in DIN.

    Input shape
        - A tensor with shape: (batch_size, ..., input_dim).

    Output shape
        - A tensor with shape: (batch_size, ..., input_dim).
    """
    def __init__(self, axis=-1, epsilon=1e-8, **kwargs):
        """

        :param axis: int axis
        :param epsilon: float 参数(见论文)
        :param kwargs:
        """
        super(Dice, self).__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
        # inside params
        self.bn = None
        self.alphas = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(Dice, self).build(input_shape)
        self.bn = layers.BatchNormalization(axis=self.axis, epsilon=self.epsilon, center=False, scale=False)
        self.alphas = self.add_weight(name='dice_alphas', shape=(input_shape[-1],),
                                      trainable=True, initializer='zeros')

    def call(self, inputs, training=None, **kwargs):
        inputs_norm = self.bn(inputs, training=training)
        x_p = tf.sigmoid(inputs_norm)

        return self.alphas * (1.0 - x_p) * inputs + x_p * inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
        }
        base_config = super(Dice, self).get_config()
        base_config.update(config)
        return base_config


def activation_layer(activation):
    """
    激活层

    :param activation: str 激活函数名
    :return:
    """
    # 1,激活层
    if activation == 'Dice' or activation == 'dice':
        act_layer = Dice()
    elif isinstance(activation, str):
        act_layer = tf.keras.layers.Activation(activation)
    elif isinstance(activation, tf.keras.layers.Layer):
        act_layer = activation()
    else:
        raise ValueError('Invalid type for activation in activation_layer.')

    return act_layer


