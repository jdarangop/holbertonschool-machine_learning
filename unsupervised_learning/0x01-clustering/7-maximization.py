#!/usr/bin/env python3
""" Maximization """
import numpy as np


def maximization(X, g):
    """ calculates the maximization step in the EM algorithm for a GMM.
        Args:
            X: (numpy.ndarray) containing the data set.
            g: (numpy.ndarray) containing the posterior probabilities
                                for each data point in each cluster.
        Returns: pi, m, S
            pi: (numpy.ndarray) containing the updated priors
                                for each cluster.
            m: (numpy.ndarray) containing the updated centroid
                               means for each cluster.
            S: (numpy.ndarray) containing the updated covariance
                               matrices for each cluster.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(g) != np.ndarray or len(g.shape) != 2:
        return None, None, None
    n, d = X.shape
    k = g.shape[0]

    pi = np.sum(g, axis=1) / n

    m = np.zeros((k, d))
    S = np.zeros((k, d, d))
    for i in range(k):
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        S[i] = np.matmul(g[i] * (X - m[i]).T, (X - m[i])) / np.sum(g[i])

    return pi, m, S
