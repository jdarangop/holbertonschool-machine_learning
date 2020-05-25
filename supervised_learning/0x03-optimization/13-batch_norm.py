#!/usr/bin/env python3
""" Batch Normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output of a
        neural network using batch normalization.
        Z: (numpy.ndarray) Array to be normalized.
        gama: (numpy.ndarray) with the scales.
        beta: (numpy.ndarray) with th offsets.
        epsilon: (numpy.ndarray) to avoid zero division.
        Returns: the normalized z matrix.
    """
    mean = np.sum(Z, axis=0) / Z.shape[0]
    sd = np.sum((Z - mean) ** 2, axis=0) / Z.shape[0]
    Znorm = (Z - mean) / np.sqrt(sd + epsilon)
    return (gamma * Znorm) + beta
