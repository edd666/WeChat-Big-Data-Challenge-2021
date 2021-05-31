# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-12
# @Contact : liaozhi_edo@163.com


"""
    layer utils
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers


class NoMask(layers.Layer):
    """
    去掉mask
    """
    def __init__(self, **kwargs):
        super(NoMask, self).__init__(**kwargs)

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(NoMask, self).build(input_shape)

    def call(self, x, mask=None, **kwargs):
        return x

    def compute_mask(self, inputs, mask=None):
        return None


def concat_func(inputs, axis=-1, mask=False):
    """
    将输入(tensor)合并

    :param inputs: list 输入tensor
    :param axis: int axis for concat
    :param mask: bool 是否支持mask
    :return:
    """
    if not mask:
        inputs = list(map(NoMask(), inputs))

    if len(inputs) == 1:
        return inputs[0]
    else:
        return tf.keras.layers.Concatenate(axis=axis)(inputs)



