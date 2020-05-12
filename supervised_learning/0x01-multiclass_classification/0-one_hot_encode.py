#!/usr/bin/env python3
""" One Hot Encode """
import numpy as np


def one_hot_encode(Y, classes):
    """ Encode some categorical values to one hot
        Y: numpy ndarray with categorical values.
        classes: is the maximum number of classes found in Y
        Return: numpy ndarray with one-hot matrix.
    """
    if any(map(lambda i: i < 0, Y)) or Y is None or classes < 0:
        return None
    result = np.zeros((classes, len(Y)))
    counter = 0
    for i in Y:
        result[i, counter] += 1
        counter += 1
    return result
