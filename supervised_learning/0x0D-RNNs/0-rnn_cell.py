#!/usr/bin/env python3
""" RNNCell """
import numpy as np


class RNNCell(object):
    """ RNNCell class """

    def __init__(self, i, h, o):
        """ Initializer. """
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ performs forward propagation for one time step.
            Args:
                h_prev: (numpy.ndarray) contains the data input for the cell.
                x_t: (numpy.ndarray) containing the previous hidden state.
            Returns:
                h_next: is the next hidden state.
                y is the output of the cell.
        """
        X = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(X, self.Wh) + self.bh)
        z = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

        return h_next, y
