#!/usr/bin/env python3
""" Adam """
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon,
                          var, grad, v, s, t):
    """ updates a variable in place using
        the Adam optimization algorithm.
        alpha: (float) the learning rate.
        beta1: (float) the weight used for the first moment.
        beta2: (float) the weight used for the second moment.
        epsilon: (float) small number to avoid zero division.
        var: (numpy.ndarray) the variable to be updated.
        grad: (numpy.ndarray) the gradient of var.
        v: (numpy.ndarray) the previous first moment of var.
        s: (numpy.ndarray) the previous second moment of var.
        t: (int) the time step used for bias correction.
        Returns: the updated variable, the new first momment,
                 and the new second moment.
    """
    dV = (beta1 * v) + ((1 - beta1) * grad)
    dS = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    dV_corr = dV / (1 - (beta1 ** t))
    dS_corr = dS / (1 - (beta2 ** t))
    var = var - (alpha * (dV_corr / (np.sqrt(dS_corr) + epsilon)))
    return var, dV, dS
