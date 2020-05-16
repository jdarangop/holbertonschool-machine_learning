#!/usr/bin/env python3
""" Loss """
import tensorflow as tf


def calculate_loss(y, y_pred):
    """ Calculates the softmax cross-entropy loss.
        y: (tf.placeholder) with labels of the input data.
        y_pred: (tf.tensor) with network's prediction.
        Returns: (tf.tensor) containing the loss of the prediction.
    """
    return tf.losses.softmax_cross_entropy(y, logits=y_pred)
