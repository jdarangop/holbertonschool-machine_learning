#!/usr/bin/env python3
""" RMSProp Upgraded """
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ creates the training operation for a
        neural network in tensorflow using the
        RMSProp optimization algorithm.
        loss: (tf.Tensor) the loss of the network.
        alpha: (float) the learning rate.
        beta2: (float) the RMSProp weight.
        epsilon: (float) a small number to avoid zero division.
        Returns: the RMSProp optimization operation.
    """
    return tf.train.RMSPropOptimizer(alpha, beta2,
                                     epsilon=epsilon).minimize(loss)
