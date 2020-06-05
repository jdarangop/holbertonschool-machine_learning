#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """ performs a valid convolution on grayscale images.
        Args:
            images: (numpy.ndarray) containing multiple grayscale images.
            kernel: (numpy.ndarray) containing the kernel for the convolution.
        Returns:
            (numpy.ndarray) containing the convoluded images.
    """
    number_img = images.shape[0]
    img_row = images.shape[1]
    img_col = images.shape[2]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
    s_row = stride[0]
    s_col = stride[1]
    if padding == 'valid':
        pad_row = 0
        pad_col = 0
    elif padding == 'same':
        pad_row = int(((img_row - 1) * s_row + kernel_row - img_row) / 2) + 1
        pad_col = int(((img_col - 1) * s_col + kernel_col - img_col) / 2) + 1
    elif type(padding) == tuple:
        pad_row = padding[0]
        pad_col = padding[1]
    images_pad = np.pad(images, ((0, 0), (pad_row, pad_row),
                                 (pad_col, pad_col)),
                        'constant', constant_values=(0))
    img_p_row = images_pad.shape[1]
    img_p_col = images_pad.shape[2]
    result = np.zeros((number_img, int((img_p_row - kernel_row) / s_row) + 1,
                       int((img_p_col - kernel_col) / s_col) + 1))
    h = int((img_p_row - kernel_row) / s_row) + 1
    w = int((img_p_col - kernel_col) / s_col) + 1
    for j in range(h):
        for i in range(w):
            result[:, j, i] = np.sum(images_pad[:,
                                     j * s_row:j * s_row + kernel_row,
                                     i * s_col:i * s_col + kernel_col] *
                                     kernel,
                                     axis=(1, 2))
    return result
