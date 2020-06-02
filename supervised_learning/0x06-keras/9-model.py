#!/usr/bin/env python3
""" Save and Load Model """
import tensorflow.keras as K


def save_model(network, filename):
    """ saves the entire model.
        Args:
            network: (tf.Tensor) model to save.
            filename: (str) path where the model should be saved.
        Returns:
            None
    """
    network.save(filename)


def load_model(filename):
    """ loads an entire model.
        Args:
            filename: (str) path where the model is.
        Returns:
            (tf.Tensor) the loaded model.
    """
    return K.models.load_model(filename)
