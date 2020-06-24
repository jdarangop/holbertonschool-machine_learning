#!/usr/bin/env python3
""" Inception Block """
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ build an inception block
        Args:
            A_prev: (tf.Tensor) previous layer.
            filters: (tuple) containing the number of filters.
        Returns:
            the concatenated output of the inception block.
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    conv1 = K.layers.Conv2D(F1, (1, 1), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(A_prev)
    conv3R = K.layers.Conv2D(F3R, (1, 1), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(A_prev)
    conv3 = K.layers.Conv2D(F3,  (3, 3), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(conv3R)
    conv5R = K.layers.Conv2D(F5R, (1, 1), padding='same',
                             kernel_initializer='he_normal',
                             activation='relu')(A_prev)
    conv5 = K.layers.Conv2D(F5, (5, 5), padding='same',
                            kernel_initializer='he_normal',
                            activation='relu')(conv5R)
    maxpool = K.layers.MaxPooling2D((3, 3), strides=(1, 1),
                                    padding='same')(A_prev)
    convmax = K.layers.Conv2D(FPP, (1, 1), padding='same',
                              kernel_initializer='he_normal',
                              activation='relu')(maxpool)

    incep_block = K.layers.concatenate([conv1, conv3,
                                        conv5, convmax])
    return incep_block
