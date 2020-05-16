#!/usr/bin/env python3
""" Accuracy """
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculate the accuracy of a prediction.
        y: (tf.placeholder) labels of the input data.
        y_pred: (tf.tensor) network's predictions.
        Returns: (tf.tensor) decimal accuracy of the prediction.
    """
    y_max = tf.argmax(y_pred, axis=1)
    y_pred_max = tf.argmax(y_pred, axis=1)
    bias = tf.cast(tf.equal(y_max, y_pred_max), dtype=tf.float32)
    return tf.reduce_mean(bias)
