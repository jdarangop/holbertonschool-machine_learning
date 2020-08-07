#!/usr/bin/env python3
""" Continuous Posterior """
from scipy import special


def posterior(x, n, p1, p2):
    """ calculates the posterior probability of obtaining the data.
        Args:
            x: (int) number of patients.
            n: (int) total number of patients observed.
            P: (numpy.ndarray) containing the various
               hypothetical probabilities.
            Pr: (numpy.ndarray) containing the prior beliefs of P.
        Returns:
            (numpy.ndarray)  the posterior probability of each
                             probability in P given x and n.
    """
    if type(n) != int or n <= 0:
        raise ValueError('n must be a positive integer')
    if type(x) != int or x < 0:
        err = 'x must be an integer that is greater than or equal to 0'
        raise ValueError(err)
    if x > n:
        raise ValueError('x cannot be greater than n')
    if type(p1) != float or p1 < 0 or p1 > 1:
        raise ValueError('p1 must be a float in the range [0, 1]')
    if type(p2) != float or p2 < 0 or p2 > 1:
        raise ValueError('p2 must be a float in the range [0, 1]')
    if p2 <= p1:
        raise ValueError('p2 must be greater than p1')
