#!/usr/bin/env python3
""" GMM """
import numpy as np
import sklearn.mixture


def gmm(X, k):
    """ calculates a GMM from a dataset.
        Args:
            X: (numpy.ndarray) containing the dataset.
            k: (int) the number of clusters.
        Returns:
            pi: (numpy.ndarray) containing the cluster priors.
            m: (numpy.ndarray) containing the centroid means.
            S: (numpy.ndarray) containing the covariance matrices.
            clss: (numpy.ndarray) containing the cluster indices
                                  for each data point.
            bic: (numpy.ndarray) containing the BIC value for
                                 each cluster size tested.
    """
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)
    return g.weights_, g.means_, g.covariances_, g.predict(X), g.bic(X)
