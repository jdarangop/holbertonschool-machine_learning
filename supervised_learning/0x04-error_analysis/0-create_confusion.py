#!/usr/bin/env python3
""" Create Confusion """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix.
        labels: (numpy.ndarray) one-hot matrix containing the correct labels.
        logits: (numpy.ndarray) one-hot matrix containing the predicted labels.
        Returns: (numpy.ndarray) with the confusion matrix.
    """
    classes = labels.shape[1]
    result = np.zeros((classes, classes))
    # correct = np.argmax(labels, axis=1)
    # predicted = np.argmax(logits, axis=1)
    for i in range(labels.shape[0]):
        row = np.argmax(labels[i])
        column = np.argmax(logits[i])
        result[row, column] += 1
    return result
