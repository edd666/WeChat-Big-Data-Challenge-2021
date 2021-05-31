# -*- coding: utf-8 -*- 
# @Author  : liaozhi
# @Time    : 2021-04-06
# @Contact : liaozhi_edo@163.com


"""
    Recurrent Neural Network(RNN) Cell
"""

# packages
import tensorflow as tf
from tensorflow.keras import layers
from deeprec.layers.activation import activation_layer


class AGRUCell(layers.Layer):
    """
    Attention based GRU uses the attention score to replace the update gate of GRU,
    and changes the hidden state directly.

    Input shape
        - a list of two tensor [x, att_score]
        - x is a 2D tensor with the shape: (batch_size, embedding_size), indicate x_t
        - att_score is a 2D tensor with the shape: (batch_size, 1), indicate attention score at t

    Output shape
        - a list of two tensor [output, new_states]
        - output is a 2D tensor with the shape: (batch_size, units), indicate y_t
        - new_states is a 2D tensor with the shape: (batch_size, units), indicate h_t+1

    References:
        - Deep Interest Evolution Network for Click-Through Rate Prediction

    Notice:
        - output = new_states
    """
    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros', **kwargs):
        """

        :param units: int the dim of hidden state
        :param activation: str Activation function for input step
        :param recurrent_activation: str Activation function for the recurrent step
        :param kernel_initializer: str Initializer for the kernel weights matrix in input step
        :param recurrent_initializer: str Initializer for the kernel weights matrix in recurrent step
        :param bias_initializer: Initializer for the bias
        :param kwargs:
        """
        super(AGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # inside params
        # recurrent step
        self.recurrent_kernel = None
        self.recurrent_bias = None
        self.recurrent_activation_layer = None
        # input step
        self.kernel = None
        self.bias = None
        self.activation_layer = None

    @property
    def output_size(self):
        return self.units

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(AGRUCell, self).build(input_shape)

        # recurrent step, deal previous hidden state h, include reset gate and input
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, 2 * self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )
        self.recurrent_bias = self.add_weight(
            name='recurrent_bias',
            shape=(2 * self.units,),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.recurrent_activation_layer = activation_layer(self.recurrent_activation)

        # input step, deal current input x, include reset gate and input
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[0][-1], 2 * self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(2 * self.units,),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.activation_layer = activation_layer(self.activation)
        self.built = True

    def call(self, inputs, states=None, **kwargs):
        # x is the current input, att_score is attention score
        x, att_score = inputs
        # h is previous hidden state
        h = states[0] if tf.nest.is_nested(states) else states

        # Notice: x_r represent the linear mapping from x to r(reset gare)
        # input step
        x_r = tf.nn.bias_add(
            tf.tensordot(x, self.kernel[:, 0: self.units], axes=(-1, 0)),
            self.bias[0: self.units]
        )
        x_h = tf.nn.bias_add(
            tf.tensordot(x, self.kernel[:, self.units:], axes=(-1, 0)),
            self.bias[self.units:]
        )

        # recurrent step
        h_r = tf.nn.bias_add(
            tf.tensordot(h, self.recurrent_kernel[:, 0: self.units], axes=(-1, 0)),
            self.recurrent_bias[0: self.units]
        )

        # reset gate
        r = self.recurrent_activation_layer(x_r + h_r)

        # input date
        h_h = tf.nn.bias_add(
            tf.tensordot(r * h, self.recurrent_kernel[:, self.units:], axes=(-1, 0)),
            self.recurrent_bias[self.units:]
        )
        h_hat = self.activation_layer(x_h + h_h)

        # new hidden state
        z = att_score  # replace update gate with att_score
        new_h = (1 - z) * h + z * h_hat
        new_states = [new_h] if tf.nest.is_nested(states) else new_h

        return new_h, new_states

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'kernel_initializer': self.kernel_initializer,
            'recurrent_initializer': self.recurrent_initializer,
            'bias_initializer': self.bias_initializer,
        }
        basic_config = super(AGRUCell, self).get_config()
        basic_config.update(config)

        return basic_config


class AUGRUCell(layers.Layer):
    """
    GRU with attentional update gate uses update gate * attention score to replace the update gate of GRU.

    Input shape
        - a list of two tensor [x, att_score]
        - x is a 2D tensor with the shape: (batch_size, embedding_size), indicate x_t
        - att_score is a 2D tensor with the shape: (batch_size, 1), indicate attention score at t

    Output shape
        - a list of two tensor [output, new_states]
        - output is a 2D tensor with the shape: (batch_size, units), indicate y_t
        - new_states is a 2D tensor with the shape: (batch_size, units), indicate h_t+1

    References:
        - Deep Interest Evolution Network for Click-Through Rate Prediction

    Notice:
        - output = new_states
    """
    def __init__(self, units, activation='tanh', recurrent_activation='sigmoid', kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros', **kwargs):
        """

        :param units: int the dim of hidden state
        :param activation: str Activation function for input step
        :param recurrent_activation: str Activation function for the recurrent step
        :param kernel_initializer: str Initializer for the kernel weights matrix in input step
        :param recurrent_initializer: str Initializer for the kernel weights matrix in recurrent step
        :param bias_initializer: Initializer for the bias
        :param kwargs:
        """
        super(AUGRUCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.recurrent_activation = recurrent_activation
        self.kernel_initializer = kernel_initializer
        self.recurrent_initializer = recurrent_initializer
        self.bias_initializer = bias_initializer

        # inside params
        # recurrent step
        self.recurrent_kernel = None
        self.recurrent_bias = None
        self.recurrent_activation_layer = None
        # input step
        self.kernel = None
        self.bias = None
        self.activation_layer = None

    @property
    def output_size(self):
        return self.units

    @property
    def state_size(self):
        return self.units

    def build(self, input_shape):
        # Be sure to call this somewhere!
        super(AUGRUCell, self).build(input_shape)

        # recurrent step, deal previous hidden state h, include reset gate, update gate, and input
        self.recurrent_kernel = self.add_weight(
            name='recurrent_kernel',
            shape=(self.units, 3 * self.units),
            initializer=self.recurrent_initializer,
            trainable=True,
        )
        self.recurrent_bias = self.add_weight(
            name='recurrent_bias',
            shape=(3 * self.units,),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.recurrent_activation_layer = activation_layer(self.recurrent_activation)

        # input step, deal current input x, include reset gate, update gate, and input
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[0][-1], 3 * self.units),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(3 * self.units,),
            initializer=self.bias_initializer,
            trainable=True,
        )
        self.activation_layer = activation_layer(self.activation)
        self.built = True

    def call(self, inputs, states=None, training=None, **kwargs):
        # x is the current input, att_score is attention score
        x, att_score = inputs
        # h is previous hidden state
        h = states[0] if tf.nest.is_nested(states) else states

        # Notice: x_r represent the linear mapping from x to r(reset gare)
        # input step
        x_r = tf.nn.bias_add(
            tf.tensordot(x, self.kernel[:, 0: self.units], axes=(-1, 0)),
            self.bias[0: self.units]
        )
        x_z = tf.nn.bias_add(
            tf.tensordot(x, self.kernel[:, self.units: 2 * self.units], axes=(-1, 0)),
            self.bias[self.units: 2 * self.units]
        )
        x_h = tf.nn.bias_add(
            tf.tensordot(x, self.kernel[:, 2 * self.units:], axes=(-1, 0)),
            self.bias[2 * self.units:]
        )

        # recurrent step
        h_r = tf.nn.bias_add(
            tf.tensordot(h, self.recurrent_kernel[:, 0: self.units], axes=(-1, 0)),
            self.recurrent_bias[0: self.units]
        )
        h_z = tf.nn.bias_add(
            tf.tensordot(h, self.recurrent_kernel[:, self.units: 2 * self.units], axes=(-1, 0)),
            self.recurrent_bias[self.units: 2 * self.units]
        )

        # reset gate and update gate
        r = self.recurrent_activation_layer(x_r + h_r)
        z = self.recurrent_activation_layer(x_z + h_z)

        # input date
        h_h = tf.nn.bias_add(
            tf.tensordot(r * h, self.recurrent_kernel[:, 2 * self.units:], axes=(-1, 0)),
            self.recurrent_bias[2 * self.units:]
        )
        h_hat = self.activation_layer(x_h + h_h)

        # new hidden state
        z = att_score * z
        new_h = (1 - z) * h + z * h_hat
        new_states = [new_h] if tf.nest.is_nested(states) else new_h

        return new_h, new_states

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'recurrent_activation': self.recurrent_activation,
            'kernel_initializer': self.kernel_initializer,
            'recurrent_initializer': self.recurrent_initializer,
            'bias_initializer': self.bias_initializer,
        }
        basic_config = super(AUGRUCell, self).get_config()
        basic_config.update(config)

        return basic_config
