#!/usr/bin/env python3
""" Likelihood """
import numpy as np
import math


def intersection(x, n, P, Pr):
    """ calculates the intersection of obtaining this
        data with the various hypothetical probabilities.
        Args:
            x: (int) number of patients.
            n: (int) total number of patients observed.
            P: (numpy.ndarray) containing the various
               hypothetical probabilities.
            Pr: (numpy.ndarray) containing the prior beliefs of P.
        Returns:
            (numpy.ndarray) containing the intersection of
                            obtaining x and n with each
                            probability in P.
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if type(Pr) != np.ndarray or P.shape != Pr.shape:
        raise TypeError('Pr must be a numpy.ndarray with the same shape as P')
    for i, j in zip(P, Pr):
        if i < 0 or i > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
        if j < 0 or j > 1:
            raise ValueError('All values in Pr must be in the range [0, 1]')
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError('Pr must sum to 1')
    result = np.zeros(P.shape)
    comb = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))
    for i in range(P.shape[0]):
        result[i] = comb * (P[i] ** x) * ((1 - P[i]) ** (n - x)) * Pr[i]
    return result
