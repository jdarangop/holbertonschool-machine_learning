#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


def convolution(image, kernel):
    """ perform the convolution for an image.
        Args:
            image: (numpy.ndarray) image which the
                   convolution should be applied.
            kernel: (numpy.ndarray) kernel to perform
                    the convolution.
        Returns:
            (numpy.ndarray) with the result of the convolution
                            in the image.
    """
    img_row = image.shape[0]
    img_col = image.shape[1]
    kernel_side = kernel.shape[0]
    # col_k = kernel.shape[1]
    result = np.zeros((img_row - kernel_side + 1, img_col - kernel_side + 1))
    i = 0
    j = 0
    while(True):
        result[j][i] += np.sum(image[j:j + kernel_side,
                               i:i + kernel_side] * kernel)
        if i < img_col - kernel_side:
            i += 1
        elif j < img_row - kernel_side:
            j += 1
            i = 0
        else:
            return result


def pad_with(vector, pad_width, iaxis, kwargs):
    """pad_with function from numpy.pad documentation"""
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    return vector


def convolve_grayscale_same(images, kernel):
    """ performs a valid convolution on grayscale images keeping dim.
        Args:
            images: (numpy.ndarray) containing multiple grayscale images.
            kernel: (numpy.ndarray) containing the kernel for the convolution.
        Returns:
            (numpy.ndarray) containing the convoluded images.
    """
    images_convoluded = []
    for img in images:
        padding = int((kernel.shape[0] - 1) / 2)
        # cover = np.zeros(i.shape[0] + (2 * padding))
        # img_pad = np.pad(img, padding, pad_with)
        img_pad = np.zeros((img.shape[0] + (2 * padding),
                            img.shape[1] + (2 * padding)))
        img_pad[padding:padding + img.shape[0],
                padding:padding + img.shape[1]] = img
        images_convoluded.append(convolution(img_pad, kernel))
    return np.array(images_convoluded)
