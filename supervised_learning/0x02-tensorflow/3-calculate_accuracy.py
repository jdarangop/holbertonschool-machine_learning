#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculate the accuracy of a prediction.
        y: (tf.placeholder) labels of the input data.
        y_pred: (tf.tensor) network's predictions.
        Returns: (tf.tensor) decimal accuracy of the prediction.
    """
    sub = tf.subtract(y_pred, y)
    absolute = tf.abs(sub)
    return tf.reduce_mean(absolute)
