#!/usr/bin/env python3
""" Optimize k """
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance.
        Args:
            X: (numpy.ndarray) containing the data set.
            kmin: (int) containing the minimum number of clusters to check for.
            kmax: (int) containing the maximum number of clusters to check for.
            iterations: (int) containing the maximum number
                              of iterations for K-means.
        Returns:
            (list) containing the outputs of K-means for each cluster size.
            (list) containing the difference in variance from the
                   smallest cluster size for each cluster size.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None, None
    if type(iterations) != int or iterations <= 0:
        return None, None
    if kmin is not None and (type(kmin) != int or kmin <= 0):
        return None, None
    if kmax is not None and (type(kmax) != int or kmax <= 0):
        return None, None
    if kmax is not None and kmax <= kmin:
        return None, None

    results = []
    d_vars = []
    if kmax is None:
        end = X.shape[0]
    else:
        end = kmax + 1
    for i in range(kmin, end):
        C, clss = kmeans(X, i, iterations)
        var = variance(X, C)
        if i == kmin:
            first_var = variance(X, C)
        results.append((C, clss))
        d_vars.append(first_var - var)

    return results, d_vars
