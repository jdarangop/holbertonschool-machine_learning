#!/usr/bin/env python3
""" RNN """
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ performs forward propagation for a simple RNN.
        Args:
            rnn_cell: (RNNCell) instance.
            X: (numpy.ndarray) data to be used (t, m, i)
                t: maximum number of time steps.
                m: batch size.
                i: dimensionality of the data.
            h_0: (numpy.ndarray) initial hidden state.
        Returns:
            H: (numpy.ndarray) containing all of the hidden states.
            Y: (numpy.ndarray) containing all of the outputs.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    Y = []
    counter = 0
    for j in range(t):
        x_0 = X[j, :, :]
        h_iter = np.zeros((l, m, h))
        for k in range(l):
            h_prev, y = rnn_cells[k].forward(H[counter][k], x_0)
            x_0 = h_prev
            h_iter[k] = h_prev
        H[j + 1] = h_iter
        Y.append(y)
        counter += 1
    Y = np.array(Y)

    return H, Y
