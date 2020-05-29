#!/usr/bin/env python3
""" L2 Regularization Cost """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization
        cost: (numpy.ndarray) the cost without regularization.
        lambtha: (float) regularization parameter.
        weights: (dict)(numpy.ndarray) with the weights and biases.
        L: (int) number of layers in the neural netwrok.
        m: (int) number of data points used.
        Returns: the cost of the network with L2 regularization.
    """
    total = 0
    for i in range(1, L + 1):
        total += np.linalg.norm(weights["W" + str(i)])
    return cost + ((lambtha * total) / (2 * m))
