#!/usr/bin/env python3
""" Moving Average """
import numpy as np


def moving_average(data, beta):
    """ calculates the weighted moving average of a data set.
        data: (list) list of data to calculate the moving average of.
        beta: (float) weight used for the moving average.
        Returns: a list containing the moving averages of data.
    """
    V = 0
    result = []
    for i in range(len(data)):
        V = (beta * V) + ((1 - beta) * data[i])
        bias_corr = (1 - beta ** (i + 1))
        result.append(V / bias_corr)
    return result
