#!/usr/bin/env python3
""" Initialize GMM """
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model.
        Args:
            X: (numpy.ndarray) containing the data set.
            k: (int) containing the number of clusters.
        Returns:
            pi: (numpy.ndarray) containing the priors for each
                                cluster, initialized evenly.
            m: (numpy.ndarray) containing the centroid means for
                               each cluster, initialized with K-means.
            S: (numpy.ndarray) containing the covariance matrices for
                               each cluster, initialized as
                               identity matrices.
    """
    if type(X) != np.ndarray or type(k) != int or k <= 0:
        return None, None, None
    if len(X.shape) != 2 or k >= X.shape[0]:
        return None, None, None

    n, d = X.shape
    pi = np.tile(1 / k, (k, ))
    m, _ = kmeans(X, k)
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
