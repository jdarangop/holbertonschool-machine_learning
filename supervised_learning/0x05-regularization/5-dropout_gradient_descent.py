#!/usr/bin/env python3
""" Gradient Descent with Dropout """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha,
                             keep_prob, L):
    """ updates the weights of a neural network with
        Dropout regularization.
        Y: (numpy.ndarray) one-hot array with the correct labelsfor the data.
        weights: (dict)(numpy.ndarray) dict with the weights and biases.
        cache: (dict)(numpy.ndarray) dict with the outputs of each layer.
        alpha: (float) learning rate.
        keep_prob: (float) the porbability that a node will be kept.
        L: (int) number of layers of the network.
        Returns: None.
    """
    weights_copy = weights.copy()
    m = Y.shape[1]
    for i in range(1, L + 1)[::-1]:
        A = cache["A{}".format(i)]
        if i == L:
            dZ = A - Y
        else:
            dZ = np.matmul(W.T, dZ) * (1 - (A ** 2))
            dZ *= cache["D" + str(i)]
            dZ /= keep_prob
        dW = np.matmul(dZ, cache["A" + str(i - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        W = weights_copy["W" + str(i)] - (alpha * dW)
        b = weights_copy["b" + str(i)] - (alpha * db)
        weights["W" + str(i)] = W
        weights["b" + str(i)] = b
