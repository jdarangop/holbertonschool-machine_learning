#!/usr/bin/env python3
""" Entropy """
import numpy as np


def HP(Di, beta):
    """ calculates the Shannon entropy and P
        affinities relative to a data point.
        Args:
            Di: (numpy.ndarray) containing the pariwise
                distances between a data point and all
                other points except itself.
        Returns:
            (Hi, Pi)
    """
    exp = np.exp(-Di * beta)
    Pi = exp / np.sum(exp)
    Hi = -np.sum(Pi*np.log2(Pi))
    return Hi, Pi
