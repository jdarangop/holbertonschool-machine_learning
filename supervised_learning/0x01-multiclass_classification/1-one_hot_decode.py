#!/usr/bin/env python3
""" One Hot Decode """
import numpy as np


def one_hot_decode(one_hot):
    """ Decode a one hot
        one_hot: numpy ndarray with one hot.
        Return: numpy ndarray with categorical values
    """
    if type(one_hot) != np.ndarray:
        return None
    # result = []
    result = np.zeros(one_hot.shape[1], dtype=np.int32)
    for i in range(one_hot.shape[1]):
        if any(np.where((one_hot[:, i] != 0) |
               (one_hot[:, i] != 1), False, True)):
            return None
        element = np.argmax(one_hot[:, i])
        result[i] = element
        # result.append(element)
    # result = np.ndarray((result))
    return result
