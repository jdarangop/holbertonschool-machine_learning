#!/usr/bin/env python3
""" Valid Convolution """
import numpy as np


# def convolution(images, kernel):
def convolve_grayscale_valid(images, kernel):
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
    number_img = images.shape[0]
    img_row = images.shape[1]
    img_col = images.shape[2]
    kernel_row = kernel.shape[0]
    kernel_col = kernel.shape[1]
    result = np.zeros((number_img, img_row - kernel_row + 1,
                       img_col - kernel_col + 1))
    i = 0
    j = 0
    while(True):
        result[:, j, i] = np.sum(images[:, j:j + kernel_row,
                                 i:i + kernel_col] * kernel, axis=(1, 2))
        if i < img_col - kernel_col:
            i += 1
        elif j < img_row - kernel_row:
            j += 1
            i = 0
        else:
            return result


# def convolve_grayscale_valid(images, kernel):
def convolution(images, kernel):
    """ performs a valid convolution on grayscale images.
        Args:
            images: (numpy.ndarray) containing multiple grayscale images.
            kernel: (numpy.ndarray) containing the kernel for the convolution.
        Returns:
            (numpy.ndarray) containing the convoluded images.
    """
    # images_convoluded = []
    # for i in images:
    #     images_convoluded.append(convolution(i, kernel))
    images_conv = convolution(images, kernel)
    return images_conv
