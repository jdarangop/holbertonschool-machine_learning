#!/usr/bin/env python3
""" Forward Propagation with Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward propagation using Dropout.
        X: (numpy.ndarray) containing the input data for the network.
        weights: (dict)(numpy.ndarray) dict with the
                 weights and bias of the NN.
        L: (int) number of layers in the layers.
        keep_prob: (float) probability that a node will be kept.
        Returns: (dict) with the outputs of each layer.
    """
    cache = {}
    cache['A0'] = X
    for i in range(L):
        keyA = "A{}".format(i + 1)
        keyb = "b{}".format(i + 1)
        keyAo = "A{}".format(i)
        keyW = "W{}".format(i + 1)
        keyD = "D{}".format(i + 1)
        z = (np.matmul(weights[keyW], cache[keyAo]) +
             weights[keyb])
        if i != L - 1:
            a = np.tanh(z)
            d = np.random.rand(a.shape[0], a.shape[1]) < keep_prob
            d = np.where(d, 1, 0)
            a_reg = a * d
            a_reg /= keep_prob
        else:
            sumatory = np.sum(np.exp(z), axis=0)
            a_reg = np.exp(z) / sumatory
        cache[keyA] = a_reg
        cache[keyD] = d

    return cache
