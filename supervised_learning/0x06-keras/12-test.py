#!/usr/bin/env python3
""" Test """
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """ test a neural network.
        Args:
            network: (tf.Tensor) network model to be tested.
            data: (numpy.ndarray) data to test de model.
            labels: (numpy.ndarray) One-Hot array with the correct labels.
            verbose: (bool) determines if the output should be printed.
        Returns:
            (numpy.ndarray) with the loss and accuracy of the model.
    """
    if verbose:
        result = network.evaluate(data, labels, verbose=1)
    else:
        result = network.evaluate(data, labels, verbose=0)
    return result
