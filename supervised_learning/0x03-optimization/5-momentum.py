#!/usr/bin/env python3
""" Momentum """
import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ updates a variable using the gradient descent
        with momentum optimization algorithm.
        alpha: (float) the learning rate.
        beta1: (float) the momentum weight.
        var: (numpy.ndarray) the variable to be updated.
        grad: (numpy.ndarray) the gradient of var.
        v: the previous first moment of var.
        Returns: the updated variable and the new moment.
    """
    dV = (beta1 * v) + ((1 - beta1) * grad)
    var = var - (alpha * dV)
    return var, dV
