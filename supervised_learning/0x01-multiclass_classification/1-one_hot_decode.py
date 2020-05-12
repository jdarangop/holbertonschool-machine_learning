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
    result = []
    for i in range(one_hot.shape[0]):
        element = np.argmax(one_hot[:, i])
        if one_hot[element, i] != 1:
            return None
        result.append(element)
    result = np.array(result)
    return result
