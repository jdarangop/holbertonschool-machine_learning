#!/usr/bin/env python3
""" Definiteness """
import numpy as np


def definiteness(matrix):
    """ calculates the minor matrix of a matrix.
        Args:
            matrix: (numpy.ndarray) matrix whose
                    definiteness should be calculated.
        Returns:
            (str) Positive definite, Positive semi-definite,
                  Negative semi-definite, Negative definite,
                  or Indefinite.
    """

    if type(matrix) != np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) < 2 or len(set(matrix.shape)) != 1:
        return None
    if np.array_equal(matrix, matrix.T) is False:
        return None
    flag = None
    var = None
    for i in range(matrix.shape[0]):
        det = np.linalg.det(matrix[:i + 1, :i + 1])
        if i % 2 == 0:
            if det > 0 and (flag is None or flag == 'pos'):
                flag = 'pos'
            elif det < 0 and (flag is None or flag == 'neg'):
                flag = 'neg'
            elif det == 0:
                var = 'semi'
            else:
                return 'Indefinite'
        else:
            if det > 0:
                continue
            elif det == 0:
                var = 'semi'
            else:
                return 'Indefinite'
    if var is None:
        if flag == 'pos':
            return 'Positive definite'
        else:
            return 'Negative definite'
    else:
        if flag == 'pos':
            return 'Positive semi-definite'
        else:
            return 'Negative semi-definite'
