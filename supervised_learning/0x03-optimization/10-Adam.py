#!/usr/bin/env python3
""" Adam Upgraded """
import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ that creates the training operation for a
        neural network in tensorflow using
        the Adam optimization algorithm.
        loss: (tf.Tensor) the loss of the network.
        alpha: (float) the learning rate.
        beta1: (float) the weight used for the first moment.
        beta2: (float) the weight used for the second moment.
        epsilon: (float) a small number to avoid zero division.
        Returns: the Adam optimization operation.
    """
    return tf.train.AdamOptimizer(learning_rate=alpha,
                                  beta1=beta1,
                                  beta2=beta2,
                                  epsilon=epsilon).minimize(loss)
