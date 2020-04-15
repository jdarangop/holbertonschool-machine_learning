#!/usr/bin/env python3
""" Cat's Got Your Tongue """
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ concatenate two arrays
        mat1: numpy ndarray
        mat2: numpy ndarray
        axis: int
    """
    return np.concatenate((mat1, mat2), axis)
