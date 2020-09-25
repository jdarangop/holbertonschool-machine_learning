#!/usr/bin/env python3
""" Multi Head Attention """
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class MultiHeadAttention """

    def __init__(self, dm, h):
        """ Initializer.
            Args:
                dm: (int) representing the dimensionality
                    of the model.
                h: (int) representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """ call method.
            Args:
                Q: (tf.Tensor) containing the input to generate
                   the query matrix.
                K: (tf.Tensor) containing the input to generate
                   the key matrix.
                V: (tf.Tensor) containing the input to generate
                   the value matrix.
                mask: is always None.
            Returns:
                output: (tf.Tensor) containing the scaled dot
                        product attention.
                weights: (tf.Tensor) containing the attention weights.
        """
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        q = tf.reshape(q, (batch, -1, self.h, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = self.Wq(K)
        k = tf.reshape(k, (batch, -1, self.h, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.Wq(V)
        v = tf.reshape(v, (batch, -1, self.h, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        out, weights = sdp_attention(q, k, v, mask)

        out = tf.transpose(out, perm=[0, 2, 1, 3])

        concat = tf.reshape(out, (batch, -1, self.dm))

        output = self.linear(concat)

        return out, weights
