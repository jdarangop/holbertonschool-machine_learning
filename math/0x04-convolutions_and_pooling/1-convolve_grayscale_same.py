#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolve_grayscale_same(images, kernel):
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
    pad_row = max(int((kernel_row - 1) / 2), int(kernel_row / 2))
    pad_col = max(int((kernel_col - 1) / 2), int(kernel_col / 2))
    # images_pad = np.zeros((number_img, img_row + (2 * padding),
    #                        img_col + (2 * padding)))
    # images_pad[:, padding:padding + img_row,
    #            padding:padding + img_col] = images[:]
    images_pad = np.pad(images, ((0, 0), (pad_row, pad_row),
                                 (pad_col, pad_col)),
                        'constant', constant_values=(0))
    result = np.zeros((number_img, img_row, img_col))
    # i = 0
    # j = 0
    # while(True):
    for j in range(img_row):
        for i in range(img_col):
            result[:, j, i] = np.sum(images_pad[:, j:j + kernel_row,
                                     i:i + kernel_col] * kernel,
                                     axis=(1, 2))
        # if i < img_col - kernel_col:
        #    i += 1
        # elif j < img_row - kernel_row:
        #    j += 1
        #    i = 0
        # else:
    return result
