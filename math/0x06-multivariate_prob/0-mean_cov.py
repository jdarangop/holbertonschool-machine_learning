#!/usr/bin/env python3
""" Mean and Covariance """
import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set.
        Args:
            X: (numpy.ndarray) the data set.
        Returns:
            (numpy.ndarray) with the mean of the data set.
            (numpy.ndarray) containing the covariance
                            matrix of the data set.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    n, d = X.shape
    if n < 2:
        raise ValueError('X must contain multiple data points')
    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
