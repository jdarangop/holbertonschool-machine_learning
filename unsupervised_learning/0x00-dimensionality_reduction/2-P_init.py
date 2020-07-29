#!/usr/bin/env python3
""" Initialize t-SNE """
import numpy as np


def P_init(X, perplexity):
    """ initializes all variables required to
        calculate the P affinities in t-SNE.
        Args:
            X: (numpy.ndarray) containing the dataset
               to be transformed by t-SNE.
            perplexity: (float) the perplexity that all
                        Gaussian distributions should have.
        Returns:
            (D, P, betas, H)
    """
    n, d = X.shape
    D = (-2 * np.matmul(X, X.T) + np.sum(X**2, axis=1)
         + np.sum(X**2, axis=1)[:, np.newaxis])
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
