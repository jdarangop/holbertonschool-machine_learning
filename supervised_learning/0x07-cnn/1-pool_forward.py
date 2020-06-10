#!/usr/bin/env python3
""" Pooling Forward Prop """
import numpy as np


def pool_forward(A_prev, kernel_shape,
                 stride=(1, 1), mode='max'):
    """ performs forward propagation over a
        pooling layer of a neural network.
        Args:
            A_prev: (numpy.ndarray) containing the output of
                    the previous layer.
            kernel_shape: (tuple) containing the size of the
                          kernel for the pooling.
            stride: (tuple) containing the strides for the pooling.
            mode: (str) indicate wheter to perform maximum of average.
        Returns:
            the output of the layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    h_out = int((h_prev - kh) / sh) + 1
    w_out = int((w_prev - kw) / sw) + 1

    result = np.zeros((m, h_out, w_out, c_prev))

    for i in range(h_out):
        for j in range(w_out):
            if mode == 'max':
                result[:, i, j, :] = np.max(A_prev[:,
                                                   i * sh:i * sh + kh,
                                                   j * sw:j * sw + kw],
                                            axis=(1, 2))
            elif mode == 'avg':
                result[:, i, j, :] = np.mean(A_prev[:,
                                                    i * sh:i * sh + kh,
                                                    j * sw:j * sw + kw],
                                             axis=(1, 2))
    return result
