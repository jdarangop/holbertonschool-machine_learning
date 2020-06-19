#!/usr/bin/env python3
""" Transition Layer """
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer.
        Args:
            X: (keras.layer) output from the previous layer.
            nb_filters: (int) number of filters.
            compression: (float) compression factor.
        Returns:
            The output of the transition layer and the number
            of filters within the output, respectively.
    """
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    num = nb_filters * compression
    layer = K.layers.Conv2D(int(num),
                            (1, 1), padding='same',
                            kernel_initializer='he_normal')(layer)
    layer = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(layer)
    return layer, num
