#!/usr/bin/env python3
""" Save and Load Weights """
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ save a model's weights.
        Args:
            network: (tf.Tensor) the model whose weights should be saved.
            filename: (str) path where the models should be saved.
            save_format: (str) format which the weights should be saved.
        Returns:
            None.
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ load model's weights.
        Args:
            network: (tf.Tensor) the model which the weights
                     should be loaded.
            filename: (str) the path where the weights
                      should be loaded.
        Returns:
            None.
    """
    network.load_weights(filename)
