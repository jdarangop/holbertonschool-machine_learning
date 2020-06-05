#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
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
    channel = images.shape[3]
    kernel_row = kernel_shape[0]
    kernel_col = kernel_shape[1]
    s_row = stride[0]
    s_col = stride[1]

    pool_row = int((img_row - kernel_row) / s_row) + 1
    pool_col = int((img_col - kernel_col) / s_col) + 1

    result = np.zeros((number_img, pool_row, pool_col, channel))

    for j in range(pool_row):
        for i in range(pool_col):
            if mode == 'max':
                result[:, j, i, :] = np.max(images[:,
                                            j * s_row:(j * s_row) +
                                            kernel_row,
                                            i * s_col:(i * s_col) +
                                            kernel_col],
                                            axis=(1, 2))
            elif mode == 'avg':
                result[:, j, i, :] = np.mean(images[:,
                                             j * s_row:(j * s_row) +
                                             kernel_row,
                                             i * s_col:(i * s_col) +
                                             kernel_col],
                                             axis=(1, 2))

    return result
