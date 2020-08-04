#!/usr/bin/env python3
""" Initialize K-means """
import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means.
        Args:
            X: (numpy.ndarray) containing the dataset that
               will be used for K-means clustering.
            k: (int) containing the number of clusters.
        Returns:
            (numpy.ndarray) containing the initialized centroids
                            for each cluster, or None on failure.
    """
    if type(X) != np.ndarray or type(k) != int or len(X.shape) != 2 or k <= 0:
        return None
    n, d = X.shape
    centroids = np.zeros((k, X.shape[1]))
    centroids = np.random.uniform(low=X.min(axis=0), high=X.max(axis=0),
                                  size=(k, d))
    return centroids
