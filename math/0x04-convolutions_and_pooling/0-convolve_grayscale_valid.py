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


def convolve_grayscale_valid(images, kernel):
    """ performs a valid convolution on grayscale images.
        Args:
            images: (numpy.ndarray) containing multiple grayscale images.
            kernel: (numpy.ndarray) containing the kernel for the convolution.
        Returns:
            (numpy.ndarray) containing the convoluded images.
    """
    images_convoluded = []
    for i in images:
        images_convoluded.append(convolution(i, kernel))
    return np.array(images_convoluded)
