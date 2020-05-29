#!/usr/bin/env python3
""" Layer with L2 Regularization """
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ create a tf.layer taht includes L2 regularization
        prev: (tf.Tensor) containing output of the previous layer.
        n: (int) number of nodes of the new layer.
        activation: (tf.nn.FUNCTION) the activation function.
        lambtyha: (float) L2 regularization parameter.
        Returns: the output of the new layer.
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=lambtha)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    new_layer = tf.layers.Dense(units=n,
                                activation=activation,
                                kernel_initializer=init,
                                kernel_regularizer=regularizer)
    return new_layer(prev)
