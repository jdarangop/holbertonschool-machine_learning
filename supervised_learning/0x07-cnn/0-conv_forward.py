#!/usr/bin/env python3
""" Convolutional Forward Propagation """
import numpy as np


def conv_forward(A_prev, W, b, activation,
                 padding="same", stride=(1, 1)):
    """ performs forward propagation over a
        convolutional layer of a neural network.
        Args:
            A_prev: (numpy.ndarray) containing the output
                    of the previous layer.
            W: (numpy.ndarray) containing the kernels
               for the convolution.
            b: (numpy.ndarray) containing the biases applied
               to the convolution.
            activation: (numpy.FUNCTION) an activation function
                        applied to the convolution.
            padding: (str) indicate the type of padding used.
            stride: (tuple) containing the strides for the convolution.
        Returns:
            the output of the convolutional layer.
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    if padding == 'same':
        pad_h = int(((h_prev * (sh - 1)) - sh + kh) / 2)
        pad_w = int(((w_prev * (sw - 1)) - sw + kw) / 2)
    elif type(padding) == tuple:
        pad_h, pad_w = padding
    else:
        pad_h = 0
        pad_w = 0
    img_pad = np.pad(A_prev, ((0, 0), (pad_h, pad_h),
                              (pad_w, pad_w), (0, 0)),
                     'constant', constant_values=(0))
    img_pad_h = img_pad.shape[1]
    img_pad_w = img_pad.shape[2]
    h_out = int((img_pad_h - kh) / sh) + 1
    w_out = int((img_pad_w - kw) / sh) + 1
    result = np.zeros((m, h_out, w_out, c_new))
    for i in range(h_out):
        for j in range(w_out):
            for k in range(c_new):
                result[:, i, j, k] = np.sum(img_pad[:,
                                                    i * sh: i * sh + kh,
                                                    j * sw: j * sw + kw] *
                                            W[:, :, :, k],
                                            axis=(1, 2, 3))
    return activation(result + b)
