#!/usr/bin/env python3
""" L2 Regularization Cost """
import tensorflow as tf


def l2_reg_cost(cost):
    """ calculates the cost of a neural network
        with L2 regularization.
        cost: (tf.Tensor) cost without L2 regularization.
        Returns: A tensor containing the cost with L2 regularization.
    """
    return cost + tf.losses.get_regularization_losses()
