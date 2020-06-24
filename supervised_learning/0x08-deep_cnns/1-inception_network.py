#!/usr/bin/env python3
""" Inception Network """
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ builds the inception network.
        Args:
            Void.
        Returns:
            the keras model.
    """
    input_lay = K.Input(shape=(224, 224, 3))
    layer = K.layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2),
                            kernel_initializer='he_normal',
                            activation='relu')(input_lay)
    layer = K.layers.MaxPool2D((3, 3), padding='same',
                               strides=(2, 2))(layer)
    layer = K.layers.Conv2D(64, (1, 1), padding='same', strides=(1, 1),
                            kernel_initializer='he_normal',
                            activation='relu')(layer)
    layer = K.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1),
                            kernel_initializer='he_normal',
                            activation='relu')(layer)
    layer = K.layers.MaxPool2D((3, 3), padding='same',
                               strides=(2, 2))(layer)
    layer = inception_block(layer, (64, 96, 128, 16, 32, 32))
    layer = inception_block(layer, (128, 128, 192, 32, 96, 64))
    layer = K.layers.MaxPool2D((3, 3), padding='same',
                               strides=(2, 2))(layer)
    layer = inception_block(layer, (192, 92, 208, 16, 48, 64))
    layer = inception_block(layer, (160, 112, 224, 24, 64, 64))
    layer = inception_block(layer, (128, 128, 256, 24, 64, 64))
    layer = inception_block(layer, (112, 144, 288, 32, 64, 64))
    layer = inception_block(layer, (256, 160, 320, 32, 128, 128))
    layer = K.layers.MaxPool2D((3, 3), padding='same',
                               strides=(2, 2))(layer)
    layer = inception_block(layer, (256, 160, 320, 32, 128, 128))
    layer = inception_block(layer, (384, 192, 384, 48, 128, 128))
    layer = K.layers.AveragePooling2D((7, 7), padding='same')(layer)
    layer = K.layers.Dropout(0.4)(layer)
    layer = K.layers.Dense(units=1000, kernel_initializer='he_normal',
                           activation='softmax')(layer)

    output = K.models.Model(inputs=input_lay, outputs=layer)

    return output
