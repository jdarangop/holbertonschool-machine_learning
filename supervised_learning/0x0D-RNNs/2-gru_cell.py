#!/usr/bin/env python3
""" GRU Cell """
import numpy as np


class GRUCell(object):
    """ GRUCell class represents a gated recurrent unit. """

    def __init__(self, i, h, o):
        """ Initializer.
            Args:
                i: dimensionality of the data.
                h: dimensionality of the hidden state.
                o: dimensionality of the outputs.
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
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
        zt = 1 / (1 + np.exp(-(np.matmul(X, self.Wz) + self.bz)))
        rt = 1 / (1 + np.exp(-(np.matmul(X, self.Wr) + self.br)))

        tmp = np.concatenate((h_prev * rt, x_t), axis=1)
        ht_hat = np.tanh(np.matmul(tmp, self.Wh) + self.bh)
        h_next = (1 - zt) * h_prev + zt * ht_hat

        yt = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(yt) / np.sum(np.exp(yt), axis=1, keepdims=True)

        return h_next, y
