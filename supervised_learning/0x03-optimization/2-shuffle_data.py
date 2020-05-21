#!/usr/bin/env python3
""" Shuffle Data """
import numpy as np


def shuffle_data(X, Y):
    """ shuffles the data points in two matrices the same way.
        X: (numpy.ndarray) first matrix.
        Y: (numpy.ndarray) second matrix.
        Returns: X and Y shuffled.
    """
    index = np.random.permutation(len(X))
    return X[index], Y[index]
