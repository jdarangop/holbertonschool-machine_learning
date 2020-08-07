#!/usr/bin/env python3
""" Expectation """
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ calculates the expectation step
        in the EM algorithm for a GMM.
        Args:
            X: (numpy.ndarray) containing the data set.
            pi: (numpy.ndarray) containing the priors for
                                each cluster.
            m: (numpy.ndarray) containing the centroid means
                               for each cluster.
            S: (numpy.ndarray) containing the covariance
                               matrices for each cluster.
        Returns:
            (numpy.ndarray) containing the posterior probabilities
                            for each data point in each cluster.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) != np.ndarray or len(pi.shape) != 1:
        return None, None
    if type(m) != np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) != np.ndarray or len(S.shape) != 3:
        return None, None
    n, d = X.shape
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    pos = np.zeros((k, n))
    for i in range(k):
        pos[i] = pi[i] * pdf(X, m[i], S[i])

    g = pos / np.sum(pos, axis=0)
    log = np.log(np.sum(pos, axis=0))

    return g, log.sum()
