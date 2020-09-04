#!/usr/bin/env python3
""" Bidirectional Cell Forward """
import numpy as np


class BidirectionalCell(object):
    """ BidirectionalCell class. """

    def __init__(self, i, h, o):
        """ Initializer.
            Args:
                i: the dimensionality of the data.
                h: the dimensionality of the hidden states.
                o: the dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(i + h + o, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ calculates the hidden state in the forward
            direction for one time step.
            Args:
                h_prev: (numpy.ndarray) contains the data input for the cell.
                x_t: (numpy.ndarray) containing the previous hidden state.
            Returns:
                h_next: is the next hidden state.
        """
        X = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(X, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """ calculates the hidden state in the backward
            direction for one time step.
            Args:
                h_next: (numpy.ndarray) contains the data input for the cell.
                x_t: (numpy.ndarray) containing the previous hidden state.
            Returns:
                h_prev: is the next hidden state.
        """
        X = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(X, self.Whb) + self.bhb)

        return h_prev
