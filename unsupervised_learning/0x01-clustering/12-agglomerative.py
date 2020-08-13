#!/usr/bin/env python3
""" Agglomerative """
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ performs agglomerative clustering on a dataset.
        Args:
            X: (numpy.ndarray) containing the dataset.
            dist: (int) the maximum cophenetic distance
                        for all clusters.
        Returns:
            (numpy.ndarray) containing the cluster indices
                            for each data point.
    """
    tmp = scipy.cluster.hierarchy.linkage(X, method='ward')
    dendro = scipy.cluster.hierarchy.dendrogram(tmp,
                                                color_threshold=dist)
    graph = plt.figure()
    graph.show()
    clss = scipy.cluster.hierarchy.fcluster(tmp,
                                            t=dist,
                                            criterion='distance')
    return clss
