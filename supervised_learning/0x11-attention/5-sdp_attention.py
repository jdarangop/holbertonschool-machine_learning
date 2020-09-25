#!/usr/bin/env python3
""" Scaled Dot Product Attention """
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ calculates the scaled dot product attention.
        Args:
            Q: (tf.Tensor) containing the query matrix.
            K: (tf.Tensor) containing the key matrix.
            V: (tf.Tensor) containing the value matrix.
            mask: (tf.Tensor) containing the optional mask,
                  or defaulted to None.
        Returns:
            output: (tf.Tensor) containing the scaled
                    dot product attention.
            weights: (tf.Tensor) containing the attention weights.
    """
    matmul = tf.linalg.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(Q)[1], tf.float32)
    scale = matmul / tf.math.sqrt(dk)
    if mask is not None:
        scale += (mask * -1e9)
    softmax = tf.nn.softmax(scale)
    output = tf.linalg.matmul(softmax, V)

    return output, softmax
