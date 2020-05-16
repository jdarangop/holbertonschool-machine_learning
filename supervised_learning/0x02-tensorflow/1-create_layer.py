#!/usr/bin/env python3
""" Create Layer """
import tensorflow as tf


def create_layer(prev, n, activation):
    """ create a layer depending its number of nodes,
        the previous output and its activation function.
        prev: (tensor) output of the previous layer.
        n: (int) number of nodes in the layer.
        activation: (tf.nn) activation function.
    """
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"),
                            name='layer')
    return layer(prev)
