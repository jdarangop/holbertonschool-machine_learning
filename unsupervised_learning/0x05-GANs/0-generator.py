#!/usr/bin/env python3
""" Generator """
import tensorflow as tf


def generator(Z):
    """ creates a simple generator network for MNIST digits.
        Args:
            Z: (tf.tensor) containing the input to the generator network.
        Returns:
            X: (tf.tensor) containing the generated image.
    """
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        layer1 = tf.keras.layers.Dense(units=128,
                                       activation='relu',
                                       name='layer_1')(Z)
        layer2 = tf.keras.layers.Dense(units=784,
                                       activation='sigmoid',
                                       name='layer_2')(layer1)
    return layer2
