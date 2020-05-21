#!/usr/bin/env python3
""" Normalize """
import numpy as np


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix.
        X: (numpy.ndarray) Matrix to normalize.
        m: mean of all features.
        s: standard deviation of all features.
        Returns: The normalized matrix.
    """
    return (X - m) / s
