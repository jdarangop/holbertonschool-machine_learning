#!/usr/bin/env python3
""" LSTM Cell """
import numpy as np


class LSTMCell(object):
    """ LSTMCell class represents a gated recurrent unit. """

    def __init__(self, i, h, o):
        """ Initializer.
            Args:
                i: dimensionality of the data.
                h: dimensionality of the hidden state.
                o: dimensionality of the outputs.
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """ performs forward propagation for one time step.
            Args:
                h_prev: (numpy.ndarray) contains the data input for the cell.
                x_t: (numpy.ndarray) containing the previous hidden state.
            Returns:
                h_next: is the next hidden state.
                c_next: is the next cell state
                y: is the output of the cell.
        """
        X = np.concatenate((h_prev, x_t), axis=1)
        ft = 1 / (1 + np.exp(-(np.matmul(X, self.Wf) + self.bf)))
        ut = 1 / (1 + np.exp(-(np.matmul(X, self.Wu) + self.bu)))
        ot = 1 / (1 + np.exp(-(np.matmul(X, self.Wo) + self.bo)))

        c_hat = np.tanh(np.matmul(X, self.Wc) + self.bc)
        c_next = ut * c_hat + ft * c_prev
        h_next = ot * np.tanh(c_next)

        yt = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(yt) / np.sum(np.exp(yt), axis=1, keepdims=True)

        return h_next, c_next, y
