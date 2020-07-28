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

    def pdf(self, x):
        """ calculates the PDF at a data point.
            Args:
                x: (numpy.ndarray) containing the data point
                   whose PDF should be calculated.
            Returns:
                (float) containing the value of the PDF.
        """
        if type(x) != np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError('x must have the shape ({d}, 1)'
                             .format(self.data[0]))
        n = x.shape[0]
        den = np.sqrt(((2 * np.pi) ** n) * np.linalg.det(self.cov))
        cov_inv = np.linalg.inv(self.cov)
        expo = (-0.5 * np.matmul(np.matmul((x - self.mean).T, cov_inv),
                                 (x - self.mean)))
        result = (1 / den) * np.exp(expo[0][0])
        return result
