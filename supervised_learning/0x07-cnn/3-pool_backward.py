#!/usr/bin/env python3
""" Pooling Backpropagation """
import numpy as np


def pool_backward(dA, A_prev, kernel_shape,
                  stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling
        layer of a neural network.
        Args:
            dA: (numpy.ndarray) containing the partial derivatives
                with respect to the output of the pooling layer.
            A_prev: (numpy.ndarray) containing the output of
                    the previous layer.
            kernel_shape: (tuple) containing the size of the
                          kernel for the pooling.
            stride: (tuple) containing the strides for the pooling.
            mode: (str) indicate whether to perform maximum or average.
        Returns:
            the partial derivatives with respect to the previous layer.
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    dA_prev = np.zeros(A_prev.shape)
    for z in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    h_init = i * sh
                    h_end = i * sh + kh
                    w_init = j * sw
                    w_end = j * sw + kw
                    if mode == 'max':
                        value = np.max(A_prev[z,
                                              h_init:h_end,
                                              w_init:w_end,
                                              k])
                        mask = np.where(A_prev[z,
                                               h_init:h_end,
                                               w_init:w_end,
                                               k] == value, 1, 0)
                        mask = mask * dA[z, i, j, k]
                    elif mode == 'avg':
                        mask = np.ones(kernel_shape) * (dA[z, i, j, k] /
                                                        (kh * kw))
                    dA_prev[z, h_init:h_end,
                            w_init:w_end, k] += mask
    return dA_prev
