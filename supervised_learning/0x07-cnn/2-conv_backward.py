#!/usr/bin/env python3
""" Convolutional Backpropagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same",
                  stride=(1, 1)):
    """ performs back propagation over a convolutional
        layer of a neural network.
        Args:
            dZ: (numpy.ndarray) containing the partial derivaties
                with respect to the unactivated output.
            A_prev: (numpy.ndarray) containing the output of
                    the previous layer.
            W: (numpy.ndarray) containing the kernels for the convolution.
            b: (numpy.ndarray) containing the biases applied
               to the convolution.
            padding: (str) indicate the type of padding used.
            stride: (tuple) containing the strides for the convolution.
        Returns:
            the partial derivaties with respect to the previous layer
            (dA_prev), the kernels(dW), and the biases (db).
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding == 'same':
        pad_h = int((h_prev * (sh - 1) + kh) / 2) + 1
        pad_w = int((w_prev * (sw - 1) + kw) / 2) + 1
    else:
        pad_h = 0
        pad_w = 0
    img_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                              (pad_w, pad_w), (0, 0)),
                     'constant', constant_values=(0))
    dA_prev = np.zeros(img_pad.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2))
    for z in range(m):
        for i in range(h_new):
            for j in range(w_new):
                for k in range(c_new):
                    dA_prev[z, i * sh:i * sh + kh,
                            j * sw:j * sw + kw, :] += (W[:, :, :, k] *
                                                       dZ[z, i, j, k])
                    dW[:, :, :, k] += (img_pad[z, i * sh:i * sh + kh,
                                               j * sw:j * sw + kw, :] *
                                       dZ[z, i, j, k])
    return dA_prev, dW, db
