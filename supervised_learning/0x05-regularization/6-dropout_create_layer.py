#!/usr/bin/env python3
""" Create a Layer with Dropout """
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates a layer of a neural network using dropout
        prev: (tf.Tensor) the output of the previous layer.
        n: (int) the number of nodes in the new layer.
        activation: (tf.nn.FUNCTION) the activation function
                    that should be used on the layer.
        keep_prob: (float) the probability that a node will be kept.
        Returns: the ouput of the new layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n,
                                activation=activation,
                                kernel_initializer=init)
    drop = tf.layers.Dropout(rate=keep_prob)
    return drop(new_layer(prev))
