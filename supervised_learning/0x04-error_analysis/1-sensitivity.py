#!/usr/bin/env python3
""" Sensitivity """
import numpy as np


def sensitivity(confusion):
    """ calculates the sensitivity for each class in confusion matrix.
        confusion: (numpy.ndarray) with a confusion matrix.
        Returns: (numpy.ndarray) containing the sensitivity of each class.
    """
    classes_num = confusion.shape[0]
    classes = range(classes_num)
    result = confusion[classes, classes] / np.sum(confusion, axis=1)
    return result
