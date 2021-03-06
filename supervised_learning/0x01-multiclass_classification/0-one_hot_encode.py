#!/usr/bin/env python3
""" One Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ Encode some categorical values to one hot
        Y: numpy ndarray with categorical values.
        classes: is the maximum number of classes found in Y
        Return: numpy ndarray with one-hot matrix.
    """
    if type(Y) != np.ndarray or type(classes) != int or classes < np.max(Y):
        return None
    result = np.zeros((classes, len(Y)))
    counter = 0
    for i in Y:
        if i < 0:
            return None
        result[i, counter] += 1
        counter += 1
    return result
