#!/usr/bin/env python3
""" Markov Chain """
import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain
        being in a particular state after a specified
        number of iterations.
        Args:
            P: (numpy.ndarray) the transition matrix.
            s: (numpy.ndarray) the probability of starting
                               in each state.
            t: (int) the number of iterations.
        Returns:
            (numpy.ndarray) the probability of being in a
                            specific state after t iterations,
                            or None on failure.
    """
    if type(P) != np.ndarray or len(P.shape) != 2:
        return None
    if type(s) != np.ndarray or len(s.shape) != 2:
        return None
    if P.shape[0] != s.shape[1] or P.shape[0] != P.shape[1]:
        return None
    if not np.isclose(np.sum(P, axis=1), np.ones((P.shape[0]))).any():
        return None
    mat_pow = np.linalg.matrix_power(P, t)
    result = np.matmul(s, mat_pow)
    return result
