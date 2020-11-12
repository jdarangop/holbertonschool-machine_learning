#!/usr/binenv python3
""" Rotate """
import tensorflow as tf


def rotate_image(image):
    """ rotates an image by 90 degrees counter-clockwise.
        Args:
            image: (tf.Tensor) containing the image to rotate.
        Returns:
            the rotated image.
    """
    return tf.image.rot90(image)
