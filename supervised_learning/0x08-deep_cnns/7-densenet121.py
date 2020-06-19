#!/usr/bin/env python3
""" DenseNet-121 """
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds the DenseNet-121 architecture.
        Args:
            growth_rate: (float) the growth rate.
            compression: (float) compression factor.
        Returns:
            the keras model.
    """
    input_lay = K.layers.Input(shape=(224, 224, 3))
    layer = K.layers.BatchNormalization()(input_lay)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same',
                            kernel_initializer='he_normal')(layer)
    layer = K.layers.MaxPool2D((3, 3), strides=(2, 2), padding='same')(layer)
    layer, nb_filters = dense_block(layer, growth_rate * 2, growth_rate, 6)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 12)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 24)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 16)
    layer = K.layers.AveragePooling2D((7, 7), padding='same')(layer)
    layer = K.layers.Dense(units=1000, activation='softmax',
                           kernel_initializer='he_normal')(layer)
    model = K.models.Model(inputs=input_lay, outputs=layer)

    return model
