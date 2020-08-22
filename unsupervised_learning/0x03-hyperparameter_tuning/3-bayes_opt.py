#!/usr/bin/env python3
""" Initialize Bayesian Optimization """
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization(object):
    """ BayesianOptimization class """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """ Initializer.
            Args:
                f: (numpy.ndarray) the black-box function to be optimized.
                X_init: (numpy.ndarray) representing the inputs already
                        sampled with the black-box function.
                Y_init: (numpy.ndarray) representing the outputs of the
                        black-box function for each input in X_init.
                bounds: (tuple) containing (min, max) representing the bounds
                        of the space in which to look for the optimal point.
                ac_sample: (int) the number of samples that should be
                           analyzed during acquisition.
                l: (int) the length parameter for the kernel.
                sigma_f: (float) the standard deviation given to the output
                         of the black-box function.
                xsi: (float) the exploration-exploitation factor
                     for acquisition.
                minimize: (bool) determining whether optimization should be
                          performed for minimization (True)
                          or maximization (False).
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples)[:, np.newaxis]
        self.xsi = xsi
        self.minimize = minimize
