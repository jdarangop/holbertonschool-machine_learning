#!/usr/bin/env python3
""" Specificity """
import numpy as np


def specificity(confusion):
    """ calculates the specificity for each class
        in a confusion matrix.
        confusion: (numpy.ndarray) confusion matrix.
        Returns: (numpy.ndarray) containing the specificity of each class.
    """
    classes = range(confusion.shape[0])
    num = (np.sum(confusion[classes]) - np.sum(confusion[classes], axis=0) -
           np.sum(confusion[classes], axis=1) + confusion[classes, classes])
    den = ((np.ones(confusion.shape[0]) * np.sum(confusion[classes])) -
           np.sum(confusion[classes], axis=1))
    return num / den
