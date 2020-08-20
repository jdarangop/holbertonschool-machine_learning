#!/usr/bin/env python3
""" Gaussian Process """
import numpy as np


class GaussianProcess(object):
    """ GaussianProcess class """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ Initializer.
            Args:
                X_init: (numpy.ndarray) representing the inputs
                        already sampled with the black-box function.
                Y_init: (numpy.ndarray) representing the outputs of the
                        black-box function for each input in X_init.
                l: (int) the length parameter for the kernel.
                sigma_f: (int) the standard deviation given to the
                         output of the black-box function.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ calculates the covariance kernel matrix between two matrices.
            Args:
                X1: (numpy.ndarray) first matrix.
                X2: (numpy.ndarray) second matrix.
            Returns:
                (numpy.ndarray) the covariance matrix.
        """
        dist = (np.sum(X1 ** 2, 1).reshape(-1, 1) +
                np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T))
        return (self.sigma_f ** 2) * (np.exp((-0.5 / self.l ** 2) * dist))
