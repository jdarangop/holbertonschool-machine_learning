#!/usr/bin/env python
""" EM """
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000,
                             tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM.
        Args:
            X: (numpy ndarray) containing the data set.
            k: (int) containing the number of cluster.
            iterations: (int) containing the maximum number
                              of iterations for the algorithm.
            tol: (float) containing tolerance of the log
                        likelihood, used to determine
                        early stopping.
            verbose: (bool) determines if you should print
                            information about the algorithm.
        Returns:
            pi: (numpy.ndarray) containing the priors
                                for each cluster.
            m: (numpy.ndarray) containing the centroid means
                               for each cluster.
            S: (numpy.ndarray) containing the covariance
                               matrices for each cluster.
            g: (numpy.ndarray) containing the probabilities for
                               each data point in each cluster.
            l: (float) the log likelihood of the model.
    """
    if type(X) != np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k < 1:
        return None, None, None, None, None
    if type(iterations) != int or iterations < 1:
        return None, None, None, None, None
    if type(tol) != float:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    n, d = X.shape
    pi, m, S = initialize(X, k)
    g, log = expectation(X, pi, m, S)
    log_prev = 0
    for i in range(iterations):
        if verbose is True and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(i, log))
        pi, m, S = maximization(X, g)
        g, log = expectation(X, pi, m, S)
        if np.abs(log - log_prev) <= tol:
            if verbose is True:
                print("Log Likelihood after {} iterations: {}"
                      .format(i + 1, log))
            break
        log_prev = log

    return pi, m, S, g, log
