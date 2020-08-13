#!/usr/bin/env python3
""" Absorbing Chains """
import numpy as np


def check(P, init):
    """ check if all the nodes are connected.
        Args:
            P: (numpy.ndarray) the transition matrix.
        Returns:
            (bool) True if their nodes are connected,
                   False if doesn't.
    """
    index = set([init])
    reviewed = set([])
    while True:
        tmp = list(index.difference(reviewed))
        if len(tmp) == 0:
            break
        for i in tmp:
            elements = np.where(P[:, i] != 0)
            reviewed.update(list(elements[0]))
        index = set(tmp)
    if len(reviewed) == P.shape[0] - 1:
        return True
    else:
        return False


def absorbing(P):
    """ determines if a markov chain is absorbing.
        Args:
            P: (numpy.ndarray) the transition matrix.
        Returns:
            (bool) True if it is absorbing, or False on failure.
    """
    if type(P) != np.ndarray or len(P.shape) != 2:
        return False
    if P.shape[0] != P.shape[1]:
        return False
    if False in np.isclose(np.sum(P, axis=1), np.ones((P.shape[0]))):
        return False
    n = P.shape[0]
    if np.array_equal(P, np.eye(n)):
        return True
    ones = np.where(P == 1)
    for i in ones[1]:
        tmp = np.sum(P[:, i])
        indexes = np.where(P[:, i] != 0)
        for j in indexes[0]:
            if check(P, j):
                return True
        # if tmp > 1:
        #    return True

    return False
