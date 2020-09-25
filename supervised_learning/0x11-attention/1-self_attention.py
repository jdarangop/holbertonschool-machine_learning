#!/usr/bin/env python3
""" Self Attention """
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class SelfAttention """

    def __init__(self, units):
        """ Initializer.
            Args:
                units: (int) representing the number of hidden
                       units in the alignment model.
        """
        super().__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """ call method.
            Args:
                s_prev: (tf.Tensor) containing the previous decoder
                        hidden state.
                hidden_states: (tf.Tensor) containing the outputs
                               of the encoder.
            Returns:
                context: (tf.Tensor) that contains the context
                         vector for the decoder.
                weights: (tf.Tensor) that contains the attention weights.
        """
        fixed_prev = tf.expand_dims(s_prev, axis=1)
        weights = self.V(tf.nn.tanh(self.W(fixed_prev) +
                                    self.U(hidden_states)))
        weights = tf.nn.softmax(weights)
        context = tf.reduce_sum(weights * hidden_states)

        return context, weights
