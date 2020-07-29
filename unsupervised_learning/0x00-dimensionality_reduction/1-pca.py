#!/usr/bin/env python3
""" PCA """
import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset.
        Args:
            X: (numpy.ndarray) dataset.
            ndim: (int) the new dimensionality of the
                  transformed X.
        Returns:
             (numpy.ndarray) containing the transformed version of X.
    """
    X_m = X - np.mean(X, axis=0)
    U, P, V = np.linalg.svd(X_m, full_matrices=False)
    # threshold = np.sum(P) * var
    # index = np.argmax(np.cumsum(P) > threshold)
    # result = np.matmul(U[:, :ndim], np.diag(P[:ndim]))
    result = np.matmul(X_m, V.T[:, :ndim])
    return result
