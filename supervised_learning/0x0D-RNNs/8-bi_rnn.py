#!/usr/bin/env python3
""" Bidirectional RNN """
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ performs forward propagation for a bidirectional RNN.
        Args:
            bi_cell: (BidirectinalCell) instance.
            X: (numpy.ndarray) the data to be used.
            h_0: (numpy.ndarray) the initial hidden state
                 in the forward direction.
            h_t: (numpy.ndarray) the initial hidden state
                 in the backward direction.
        Returns:
            H: (numpy.ndarray) containing all of the
               concatenated hidden states.
            Y: (numpy.ndarray) containing all of the outputs.
    """
    t, m, i = X.shape

    Hf = []
    Hb = []
    hf = h_0
    hb = h_t
    for j in range(t):
        hf = bi_cell.forward(hf, X[j, :, :])
        hb = bi_cell.backward(hb, X[t - 1 - j, :, :])
        Hf.append(hf)
        Hb.append(hb)

    Hf = np.array(Hf)
    Hb = np.array(Hb[::-1])
    H = np.concatenate((Hf, Hb), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
