#!/usr/bin/env python3
""" Regular Chains """
import numpy as np


def check(P):
    """ check if all the nodes are connected.
        Args:
            P: (numpy.ndarray) the transition matrix.
        Returns:
            (bool) True if their nodes are connected,
                   False if doesn't.
    """
    index = set([0])
    reviewed = set([])
    while True:
        tmp = list(index.difference(reviewed))
        if len(tmp) == 0:
            break
        for i in tmp:
            elements = np.where(P[i] != 0)
            reviewed.update(list(elements[0]))
        index = set(tmp)
    if len(reviewed) == P.shape[0]:
        return True
    else:
        return False


def regular(P):
    """ determines the steady state probabilities of a
        regular markov chain.
        Args:
            P: (numpy.ndarray) the transition matrix.
        returns:
            (numpy.ndarray) containing the steady state
                            probabilities, or None on failure.
    """
    if type(P) != np.ndarray or len(P.shape) != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if False in np.isclose(np.sum(P, axis=1), np.ones((P.shape[0]))):
        return None
    if not check(P):
        return None
    try:
        n = P.shape[0]
        det = P.T - np.eye(n)
        det[-1:, :] = np.ones((n,))
        vec = np.zeros((n, 1))
        vec[-1] = 1
        result = np.linalg.solve(det, vec)
        return result.T
    except Exception:
        return None
