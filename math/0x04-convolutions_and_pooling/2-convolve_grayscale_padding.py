#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
    i_row = img_row - kernel_row + 1
    i_col = img_col - kernel_col + 1
    # pad_row = max(int((kernel_row - 1) / 2), int(kernel_row / 2))
    # pad_col = max(int((kernel_col - 1) / 2), int(kernel_col / 2))
    pad_row = padding[0]
    pad_col = padding[1]
    images_pad = np.pad(images, ((0, 0), (pad_row, pad_row),
                                 (pad_col, pad_col)),
                        'constant', constant_values=(0))
    result = np.zeros((number_img, images_pad.shape[1] - kernel_row + 1,
                       images_pad.shape[2] - kernel_col + 1))
    h = images_pad.shape[1] - kernel_col + 1
    w = images_pad.shape[2] - kernel_row + 1
    # result = np.zeros((number_img, h, w))
    # i = 0
    # j = 0
    for j in range(h):
        for i in range(w):
            result[:, j, i] = np.sum(images_pad[:, j:j + kernel_row,
                                     i:i + kernel_col] * kernel,
                                     axis=(1, 2))
            # result[:, j, i] = np.sum(images_pad[:, j:j + kernel_row,
            #                          i:i + kernel_col] * kernel,
            #                          axis=(1, 2))

    # while(True):
    #    result[:, j, i] = np.sum(images_pad[:, j:j + kernel_row,
    #                             i:i + kernel_col] * kernel, axis=(1, 2))
    #    if i < img_col - kernel_col:
    #        i += 1
    #    elif j < img_row - kernel_row:
    #        j += 1
    #        i = 0
    #    else:
    return result
