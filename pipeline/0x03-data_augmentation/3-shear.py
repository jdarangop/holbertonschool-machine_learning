#!/usr/binenv python3
""" Shear """
import tensorflow as tf


def shear_image(image, intensity):
    """ randomly shears an image.
        Args:
            image: (tf.Tensor) containing the image to shear.
        Returns:
            the sheared image.
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(img,
                                                        intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    return tf.keras.preprocessing.image.array_to_img(sheared)
