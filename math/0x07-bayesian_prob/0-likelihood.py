#!/usr/bin/env python3
""" Likelihood """
import numpy as np
import math


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining
        this data given various hypothetical
        probabilities of developing severe
        side effects.
        Args:
            x: (int) number of patients.
            n: (int) total number of patients observed.
            P: (numpy.ndarray)  containing the various
               hypothetical probabilities
        Returns:
            (numpy.ndarray) containing the likelihood
                            of obtaining the data.
    """
    if n <= 0:
        raise ValueError('n must be a positive integer')
    if x < 0:
        raise ValueError('x must be an integer that \
                         is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(P) != np.ndarray:
        raise TypeError('P must be a 1D numpy.ndarray')
    result = np.zeros(P.shape)
    comb = math.factorial(n) / (math.factorial(x) * math.factorial(n - x))
    for i in range(P.shape[0]):
        if P[i] < 0 or P[i] > 1:
            raise ValueError('All values in P must be in the range [0, 1]')
        else:
            result[i] = comb * (P[i] ** x) * ((1 - P[i]) ** (n - x))
    return result
