#!/usr/bin/env python3
""" PDF """
import numpy as np


def pdf(X, m, S):
    """ calculates the probability density
        function of a Gaussian distribution.
        Args:
            X: (numpy.ndarray) containing the data points
                               whose PDF should be evaluated.
            m: (numpy.ndarray) containing the mean of the distribution.
            S: (numpy.ndarray) containing the covariance
                               of the distribution.
        Returns:
            (numpy.ndarray) containing the PDF values for each data point.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None
    if type(m) != np.ndarray or len(m.shape) != 1:
        return None
    if type(S) != np.ndarray or len(S.shape) != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape[0] != d or S.shape[0] != S.shape[1]:
        return None
    den = np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S))
    cov_inv = np.linalg.inv(S)
    expo = (-0.5 * np.sum(np.matmul(cov_inv, (X.T - m[:, np.newaxis])) *
                          (X.T - m[:, np.newaxis]), axis=0))
    result = (1 / den) * np.exp(expo)
    np.where(result < 1e-300, 1e-300, result)
    return result
