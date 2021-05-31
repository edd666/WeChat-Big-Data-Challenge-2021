# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-03-12
# @Contact : liaozhi_edo@163.com


"""
    基础Layer
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.activation import activation_layer


class LocalActivationUnit(layers.Layer):
    """
    The LocalActivationUnit used in DIN with which the representation of user interests
    varies adaptively given different candidate items.

    Input shape
        - A list of 2 tensor [query, keys]
        - query is a 3D tensor with shape: (batch_size, 1, embedding_size)
        - keys is a 3D tensor with shape : (batch_size, T, embedding_size)

    Output shape
        - 3D tensor with shape: (batch_size, T, 1)
    """
    def __init__(self, hidden_units=(36,), activation='Dice', dropout_rate=0, use_bn=False, **kwargs):
        """

        :param hidden_units: list 各层神经元数量
        :param activation: str 激活函数
        :param dropout_rate: float dropout rate
        :param use_bn: bool BatchNormalization
        :param kwargs:
        """
        super(LocalActivationUnit, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        # inside params
        self.dnn = None
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(LocalActivationUnit, self).build(input_shape)
        self.dnn = DNN(
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_bn=self.use_bn)

        size = 4 * int(input_shape[0][-1]) if len(self.hidden_units) == 0 else self.hidden_units[-1]
        self.kernel = self.add_weight(name='kernel', shape=(size, 1), initializer='random_normal', trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True)

    def call(self, inputs, training=None, **kwargs):
        query, keys = inputs

        keys_len = keys.get_shape()[1]
        queries = tf.keras.backend.repeat_elements(query, rep=keys_len, axis=1)

        # DNN
        dnn_input = tf.concat(
            [queries, keys, queries - keys, queries * keys], axis=-1)
        dnn_output = self.dnn(dnn_input, training=training)

        att_score = tf.nn.bias_add(tf.tensordot(
            dnn_output, self.kernel, axes=(-1, 0)), self.bias)

        return att_score

    def compute_output_shape(self, input_shape):
        return input_shape[1][:2] + (1,)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = {'activation': self.activation,
                  'hidden_units': self.hidden_units,
                  'use_bn': self.use_bn,
                  'dropout_rate': self.dropout_rate}
        base_config = super(LocalActivationUnit, self).get_config()
        base_config.update(config)

        return base_config


class DNN(layers.Layer):
    """
    The Multilayer Perceptron.

    Input shape
        - A tensor with shape: (batch_size, ..., input_dim).

    Output shape
        - A tensor with shape: (batch_size, ..., hidden_units[-1]).
    """
    def __init__(self, hidden_units, activation='relu', dropout_rate=0, use_bn=False, **kwargs):
        """

        :param hidden_units: list 各层神经元数量
        :param activation: str 激活函数
        :param dropout_rate: float dropout rate
        :param use_bn: bool BatchNormalization
        :param kwargs:
        """
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bn = use_bn
        # inside params
        self.kernels = None
        self.bias = None
        self.bn_layers = None
        self.dropout_layers = None
        self.activation_layers = None

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(DNN, self).build(input_shape)

        input_dim = input_shape[-1]
        hidden_units = [input_dim] + list(self.hidden_units)
        # a list of kernel and bias
        self.kernels = [
            self.add_weight(
                name='kernel' + str(i),
                shape=(hidden_units[i], hidden_units[i + 1]),
                initializer='random_normal',
                trainable=True,
            )
            for i in range(len(self.hidden_units))
        ]
        self.bias = [
            self.add_weight(
                name='bias' + str(i),
                shape=(self.hidden_units[i],),
                initializer='zeros',
                trainable=True
            )
            for i in range(len(self.hidden_units))
        ]

        # a list of BatchNormalization layers
        if self.use_bn:
            self.bn_layers = [layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        # a list of dropout layers
        self.dropout_layers = [layers.Dropout(self.dropout_rate) for _ in range(len(self.hidden_units))]

        # a list of activation layers
        self.activation_layers = [activation_layer(self.activation) for _ in range(len(self.hidden_units))]

    def call(self, inputs, training=None, **kwargs):
        deep_input = inputs
        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(
                tf.tensordot(deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])

            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)

            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self):
        config = {'activation': self.activation,
                  'hidden_units': self.hidden_units,
                  'use_bn': self.use_bn,
                  'dropout_rate': self.dropout_rate}

        base_config = super(DNN, self).get_config()
        base_config.update(config)

        return base_config


