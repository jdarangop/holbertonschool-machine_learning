#!/usr/binenv python3
""" Hue """
import tensorflow as tf


def change_hue(image, delta):
    """ changes the hue of an image.
        Args:
            image: (tf.Tensor) containing the image to change.
            delta: (float) the amount the hue should change.
        Returns:
            the altered image.
    """
    return tf.image.adjust_hue(image, delta)
