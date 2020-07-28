#!/usr/bin/env python3
""" MultiNormal """
import numpy as np


class MultiNormal(object):
    """ MultiNormal class """

    def __init__(self, data):
        """ Initializer """
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        d, n = data.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')
        X = data.T
        mean = np.mean(X, axis=0, keepdims=True)
        cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)
        self.mean = mean.T
        self.cov = cov
