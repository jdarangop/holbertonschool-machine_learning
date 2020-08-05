#!/usr/bin/env python3
""" Variance """
import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance for a data set.
        Args:
            X: (numpy.ndarray) containing the data set.
            C: (numpy.ndarray) containing the centroid
                               means for each cluster.
        Returns:
            (float) the total variance.
    """
    try:
        distance = np.sqrt((X[:, np.newaxis, 0] - C[:, 0])**2 +
                           (X[:, np.newaxis, 1] - C[:, 1])**2)
        min_dist = np.min(distance, axis=1)
        return np.sum(min_dist ** 2)
    except Exception as e:
        return None
