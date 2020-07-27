#!/usr/bin/env ptyhon3
""" Correlation """
import numpy as np


def correlation(C):
    """ calculates a correlation matrix.
        Args:
            C: (numpy.ndarray) containing a covariance matrix.
        Returns:
            (numpy.ndarray) containing the correlation matrix.
    """
    if type(C) != np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')
    var = np.diag(np.diag(C) ** (-1/2))
    corr = np.matmul(np.matmul(var, C), var)
    return corr
