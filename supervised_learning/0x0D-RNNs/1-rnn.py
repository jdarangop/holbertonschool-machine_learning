#!/usr/bin/env python3
""" RNN """
import numpy as np


def rnn(rnn_cell, X, h_0):
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
    H = [h_0]
    Y = []
    h = h_0
    for j in range(t):
        h, y = rnn_cell.forward(h, X[j, :, :])
        H.append(h)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)

    return H, Y
