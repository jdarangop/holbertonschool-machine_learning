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

    def predict(self, X_s):
        """ predicts the mean and standard deviation of
            points in a Gaussian process.
            Args:
                X_s: (numpy.ndarray)  containing all of the points whose mean
                                      and standard deviation should
                                      be calculated.
            Returns:
                mu: (numpy.ndarray) containing the mean for each
                                    point in X_s, respectively.
                sigma: (numpy.ndarray) containing the standard deviation
                                       for each point in X_s, respectively.
        """
        K = self.kernel(self.X, self.X)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), self.Y).T[0]
        sigma = np.diag(K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s))

        return mu_s, sigma

    def update(self, X_new, Y_new):
        """ updates a Gaussian Process.
            Args:
                X_new: (numpy.ndarray) represents the new sample point.
                Y_new: (numpy.ndarray) represents the new sample
                                       function value.
            Returns:
                None.
        """
        # d, _ = self.X.shape
        # self.X = np.insert(self.X, d, X_new).T
        # d, _ = self.Y.shape
        # self.Y = np.insert(self.Y, d, Y_new).T
        self.X = np.append(self.X, [X_new], axis=0)
        self.Y = np.append(self.Y, [Y_new], axis=0)
        self.K = self.kernel(self.X, self.X)
