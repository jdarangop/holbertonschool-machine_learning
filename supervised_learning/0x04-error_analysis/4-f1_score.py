#!/usr/bin/env python3
""" F1 Score """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix.
        confusion: (numpy.ndarray) confusion matrix.
        Returns: (numpy.ndarray) containing the F1 score of each class.
    """
    pre = precision(confusion)
    sen = sensitivity(confusion)
    result = (2 * (pre * sen) / (pre + sen))
    return result
