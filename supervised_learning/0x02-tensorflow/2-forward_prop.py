#!/usr/bin/env python3
""" Forward Propagation """
import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ calculate the forward propagation
        x: (tf.placeholder) with input data.
        layer_sizes: (list) with nodes in every layer.
        activations: (list) with activations functions in every layer.
        Returns: (tf.tensor) the prediction of the network.
    """
    create_layer = __import__('1-create_layer').create_layer
    tmp = x
    for i in range(len(layer_sizes)):
        tmp = create_layer(tmp, layer_sizes[i], activations[i])
    return tmp
