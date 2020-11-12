#!/usr/binenv python3
""" Flip """
import tensorflow as tf


def flip_image(image):
    """ flips an image horizontally.
        Args:
            image: (tf.Tensor) containing the image to flip.
        Returns:
            the flipped image.
    """
    return tf.image.flip_left_right(image)
