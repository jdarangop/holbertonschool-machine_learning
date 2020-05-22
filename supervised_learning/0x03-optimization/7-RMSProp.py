#!/usr/bin/env python3
""" RMSProp """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ updates a variable using the RMSProp optimization algorithm.
        alpha: (float) the learning rate.
        beta2: (float) the RMSProp weight.
        epsilon: (float) a small number to avoid division by zero.
        var: (numpy.ndarray) the variable to be updated.
        grad: (numpy.ndarray) the gradient of var.
        s: (numpy.ndarray9 previous second moment of var.
        Returns: the updated variable and the new moment.
    """
    dV = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var = var - (alpha * (grad / (np.sqrt(dV) + epsilon)))
    return var, dV
