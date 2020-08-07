#!/usr/bin/env python3
""" Hello, sklearn! """
import numpy as np
import sklearn.cluster


def kmeans(X, k):
    """ performs K-means on a dataset.
        Args:
            X: (numpy.ndarray) containing the dataset.
            k: (int) number of clusters.
        Returns:
            C: (numpy.ndarray) containing the centroid
                               means for each cluster.
            clss: (numpy.ndarray) containing the index of the
                                  cluster in C that each data
                                  point belongs to.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
