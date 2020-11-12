#!/usr/binenv python3
""" Brightness """
import tensorflow as tf


def change_brightness(image, max_delta):
    """ randomly changes the brightness of an image.
        Args:
            image: (tf.Tensor) containing the image to flip.
            max_delta: (float) the maximum amount the
                       image should be brightened.
        Returns:
            the altered image.
    """
    return tf.image.adjust_brightness(image, max_delta)
