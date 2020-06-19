#!/usr/bin/env python3
""" Identity Block """
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """ builds a identity block.
        Args:
            A_prev: (tf.Tensor) previous layer.
            filters: (tuple) contraining number of
                     filters in every convolution.
        Results:
            the activated output of the identity block.
    """
    F11, F3, F12 = filters

    conv11 = K.layers.Conv2D(F11, (1, 1), padding='same',
                             kernel_initializer='he_normal')(A_prev)
    batch_conv11 = K.layers.BatchNormalization()(conv11)
    relu_conv11 = K.layers.Activation('relu')(batch_conv11)
    conv3 = K.layers.Conv2D(F3, (3, 3), padding='same',
                            kernel_initializer='he_normal')(relu_conv11)
    batch_conv3 = K.layers.BatchNormalization()(conv3)
    relu_conv3 = K.layers.Activation('relu')(batch_conv3)
    conv12 = K.layers.Conv2D(F12, (1, 1), padding='same',
                             kernel_initializer='he_normal')(relu_conv3)
    batch_conv12 = K.layers.BatchNormalization()(conv12)
    add = K.layers.Add()([batch_conv12, A_prev])

    output = K.layers.Activation('relu')(add)

    return output
