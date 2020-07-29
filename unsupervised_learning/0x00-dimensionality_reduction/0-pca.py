#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset.
        Args:
            X: (numpy.ndarray) dataset.
        Returns:
             the weights matrix, W, that maintains
             var fraction of Xs original variance.
    """
    U, P, V = np.linalg.svd(X, full_matrices=False)
    threshold = np.sum(P) * var
    index = np.argmax(np.cumsum(P) > threshold)
    # result = np.matmul(U[:, :index], np.diag(P[:index]))
    # result = np.matmul(np.linalg.inv(X), tmp)
    return V.T[:, :index + 1]
