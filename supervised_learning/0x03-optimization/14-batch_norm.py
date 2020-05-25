#!/usr/bin/env python3
""" Batch Normalization Upgraded """
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ creates a batch normalization layer for a
        neural network in tensorflow.
        prev: (tf.Tensor) activated output of the previous layer.
        n: (int) number of nodes in the layer to be created.
        activation: (tf.Operation) activation function on the output layer.
        Returns: Tensor of the activated output for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n,
                            kernel_initializer=init,
                            name='layer')
    epsilon = 1e-8
    base = layer(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    mean, variance = tf.nn.moments(base, axes=[0])
    Z = tf.nn.batch_normalization(base,
                                  mean=mean,
                                  variance=variance,
                                  offset=beta,
                                  scale=gamma,
                                  variance_epsilon=epsilon)

    return activation(Z)
