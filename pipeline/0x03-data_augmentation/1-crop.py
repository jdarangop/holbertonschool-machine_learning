#!/usr/binenv python3
""" Crop """
import tensorflow as tf


def crop_image(image, size):
    """ performs a random crop of an image.
        Args:
            image: (tf.Tensor) containing the image to crop.
        Returns:
            the cropped image.
    """
    return tf.random_crop(image, size=size)
