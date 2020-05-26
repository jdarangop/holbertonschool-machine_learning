#!/usr/bin/env python3
""" Precision """
import numpy as np


def precision(confusion):
    """ calculates the precision for each class in a confusion matrix.
        confusion: (numpy.ndarray) confusion matrix.
        Returns: (numpy.ndarray) containing the precision of each class.
    """
    classes_num = confusion.shape[0]
    classes = range(classes_num)
    result = confusion[classes, classes] / np.sum(confusion, axis=0)
    return result
