#!/usr/bin/env python3
""" Save and Load Configuration """
import tensorflow.keras as K


def save_config(network, filename):
    """ saves a model's configuration in JSON format.
        Args:
            network: (tf.Tensor) model whose configuration should be saved.
            filename: (str) the path of the configuration should be saved.
        Returns:
            None.
    """
    with open(filename, 'w') as fp:
        fp.write(network.to_json())


def load_config(filename):
    """ loads model with a specific configuration.
        Args:
            filename: (str) path where the model should be loaded.
        Returns:
            (tf.Tensor) the loaded model.
    """
    with open(filename, 'r') as fp:
        return K.models.model_from_json(fp.read())
