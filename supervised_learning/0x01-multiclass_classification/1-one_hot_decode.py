#!/usr/bin/env python3
""" One Hot Decode """
import numpy as np


def one_hot_decode(one_hot):
    """ Decode a one hot
        one_hot: numpy ndarray with one hot.
        Return: numpy ndarray with categorical values
    """
    if one_hot is None:
        return None
    result = []
    for i in range(one_hot.shape[0]):
        result.append(np.argmax(one_hot[:, i]))
    result = np.array(result)
    return result
