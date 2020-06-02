#!/usr/bin/env python3
""" Sequential """
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ builds a basic neural network with Keras library.
        Args:
            nx: (int) number of input features.
            layers: (list) containing the nodes in each layer.
            activations: (list) containing the activation
                         functions in each layer.
            lambtha: (float) the L2 regularization parameter.
            keep_prob: (float) the probability that a node will
                       be kept for Dropout regularization.
        Returns:
            the keras model.
    """
    model = K.Sequential()
    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], input_dim=nx,
                      activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
        else:
            model.add(K.layers.Dropout(1 - keep_prob))
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                      kernel_regularizer=K.regularizers.l2(lambtha)))
    return model
