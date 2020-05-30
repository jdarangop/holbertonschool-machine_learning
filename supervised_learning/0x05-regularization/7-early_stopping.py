#!/usr/bin/env python3
""" Early Stopping """
import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early
        cost: (float) current valuation of th NN.
        opt_cost: (float) the lowest recorded validation cost of th NN.
        threshold: (float) the threshold used for early stopping.
        patience: (int) patience count used for early stopping.
        count: (int) how long the threshold has not been met.
        Returns: Boolean of whether th NN should stop and the updated count.
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
        if count >= patience:
            return True, count
        else:
            return False, count
    return False, count
