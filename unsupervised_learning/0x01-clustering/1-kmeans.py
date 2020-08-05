#!/usr/bin/env python3
""" K-means """
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


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset.
        Args:
            X: (numpy.ndarray) containing the dataset that
               will be used for K-means clustering.
            k: (int) containing the number of clusters.
        Returns:
            (numpy.ndarray) containing the centroid
                            means for each cluster.
            (numpy.ndarray) containing the index of the cluster
                            in C that each data point belongs to.
    """
    centroids = initialize(X, k)
    if centroids is None or type(iterations) != int or iterations <= 0:
        return None, None
    counter = 0
    for iteration in range(iterations):
        distance = np.sqrt((X[:, np.newaxis, 0] - centroids[:, 0])**2 +
                           (X[:, np.newaxis, 1] - centroids[:, 1])**2)
        clss = np.argmin(distance, axis=1)
        current_centroids = np.zeros(centroids.shape)
        for i in range(k):
            index = np.where(clss == i)
            centroids_data = X[index]
            if len(centroids_data) == 0:
                current_centroids[i] = initialize(X, 1)
            else:
                current_centroids[i] = np.mean(centroids_data, axis=0)
        if np.array_equal(current_centroids, centroids):
            break
        else:
            centroids = current_centroids

    distance = np.sqrt((X[:, np.newaxis, 0] - centroids[:, 0])**2 +
                       (X[:, np.newaxis, 1] - centroids[:, 1])**2)
    clss = np.argmin(distance, axis=1)

    return centroids, clss
