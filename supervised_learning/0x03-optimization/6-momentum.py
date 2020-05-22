#!/usr/bin/env python3
""" Momentum Upgraded """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ creates the training operation for a neural network
        in tensorflow using the gradient descent with
        momentum optimization algorithm.
        loss: (tf.Tensor) the loss of the network.
        alpha: (float) the learning rate.
        beta1: (float) the momentum weight.
        Returns: the momentum optimization operation.
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
