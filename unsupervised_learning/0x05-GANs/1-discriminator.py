#!/usr/bin/env python3
""" Discriminator """
import tensorflow as tf


def discriminator(X):
    """ creates a simple discriminator network for MNIST digits.
        Args:
            Z: (tf.tensor) containing the input to the
               discriminator network.
        Returns:
            X: (tf.tensor) containing the classification
               made by the discriminator..
    """
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        layer1 = tf.keras.layers.Dense(units=128,
                                       activation='relu',
                                       name='layer_1')(Z)
        layer2 = tf.keras.layers.Dense(units=1,
                                       activation='sigmoid',
                                       name='layer_2')(layer1)
    return layer2
