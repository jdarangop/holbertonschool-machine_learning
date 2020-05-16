#!/usr/bin/env python3
""" Train_Op """
import tensorflow as tf


def create_train_op(loss, alpha):
    """ creates the training operation for the network.
        loss: (tf.losses.softmax_cross_entropy) with
              loss of the networks prediction.
        alpha: (float) with the learning rate.
        Returns: an operation that trains the network
                 using gradient descent.
    """
    rlt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return rlt
