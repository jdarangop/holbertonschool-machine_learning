#!/usr/bin/env python3
""" Dense Block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block.
        Args:
            X: (keras.layer) output of the previous layer.
            nb_filters: (int) number of filters.
            growth_rate: (float) growth rate for dense block.
            layers: (int) number of layers in the dense block.
        Returns:
            The concatenated output of each layer within
            the Dense Block and the number of filters within
            the concatenated outputs, respectively.
    """
    for i in range(layers):
        # bottleneck
        layer = K.layers.BatchNormalization()(X)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(growth_rate * 4, (1, 1),
                                padding='same',
                                kernel_initializer='he_normal')(layer)
        layer = K.layers.BatchNormalization()(layer)
        layer = K.layers.Activation('relu')(layer)
        layer = K.layers.Conv2D(growth_rate, (3, 3),
                                padding='same',
                                # activation='relu',
                                kernel_initializer='he_normal')(layer)
        X = K.layers.concatenate([X, layer])
        nb_filters += growth_rate

    return X, nb_filters
