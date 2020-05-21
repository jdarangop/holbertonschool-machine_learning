#!/usr/bin/env python3
""" Normalization Constants """
import numpy as np


def normalization_constants(X):
    """ calculates the normalization (standardization)
        constants of a matrix.
        X: (numpy.ndarray) matrix to normalize.
        Returns: The mean and standard deviation of
                 each feature, respectively.
    """
    means = np.sum(X, axis=0) / X.shape[0]
    sd = np.sqrt(np.sum((X - means) ** 2, axis=0) / X.shape[0])
    return means, sd
