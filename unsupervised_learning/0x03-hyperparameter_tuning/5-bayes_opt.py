#!/usr/bin/env python3
""" Initialize Bayesian Optimization """
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """ calculates the next best sample location.
            Args:
                None.
            Returns:
                X_next: (numpy.ndarray) representing the next best
                        sample point.
                EI: (numpy.ndarray) containing the expected improvement
                                    of each potential sample.
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample, _ = self.gp.predict(self.gp.X)

        with np.errstate(divide='warn'):
            if self.minimize:
                mu_sample_opt = np.min(mu_sample)
                imp = mu_sample_opt - mu - self.xsi
            else:
                mu_sample_opt = np.max(mu_sample)
                imp = mu - mu_sample_opt - self.xsi
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """ optimizes the black-box function.
            Args:
                iterations: (int) the maximum number of
                            iterations to perform.
            Returns:
                X_opt: (numpy.ndarray) representing the optimal point.
                Y_opt: (numpy.ndarray) representing the optimal function value.
        """
        for i in range(iterations):
            X_next, _ = self.acquisition()
            Y_next = self.f(X_next)

            if X_next in self.gp.X:
                break
            self.gp.update(X_next, Y_next)
            X_prev = X_next
        if self.minimize:
            index = np.argmin(self.gp.Y)
        else:
            index = np.argmin(self.gp.Y)
        X_opt = self.gp.X[index]
        Y_opt = self.gp.Y[index]

        return X_opt, Y_opt
