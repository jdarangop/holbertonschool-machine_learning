#!/usr/bin/env python3
""" One Hot """
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ converts a label vector into a one-hot matrix.
        Args:
            labels: (numpy.ndarray) labels vector.
            classes: (int) number of classes.
        Returns:
            (numpy.ndarray) with the labels One-Hot encoding.
    """
    return K.utils.to_categorical(labels)
