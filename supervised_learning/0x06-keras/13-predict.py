#!/usr/bin/env python3
""" Predict """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ makes a prediction using neural network.
        Args:
            network: (tf.Tensor) network to make the prediction with.
            data: (numpy.ndarray) input data to make the prediction.
            verbose: (bool) determines if ouput should be printed
                     during prediction process.
        Returns:
            the prediction for the data.
    """
    if verbose:
        result = network.predict(data, verbose=1)
    else:
        result = network.predict(data, verbose=0)
    return result
