#!/usr/bin/env python3
""" ResNet-50 """
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture.
        Args:
            Void.
        Returns:
            the keras model.
    """
    input_lay = K.layers.Input(shape=(224, 224, 3))

    layer = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                            padding='same',
                            kernel_initializer='he_normal')(input_lay)
    layer = K.layers.BatchNormalization()(layer)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.MaxPooling2D((3, 3),
                                  padding='same',
                                  strides=(2, 2))(layer)
    layer = projection_block(layer, (64, 64, 256), 1)
    layer = identity_block(layer, (64, 64, 256))
    layer = identity_block(layer, (64, 64, 256))
    layer = projection_block(layer, (128, 128, 512))
    layer = identity_block(layer, (128, 128, 512))
    layer = identity_block(layer, (128, 128, 512))
    layer = identity_block(layer, (128, 128, 512))
    layer = projection_block(layer, (256, 256, 1024))
    layer = identity_block(layer, (256, 256, 1024))
    layer = identity_block(layer, (256, 256, 1024))
    layer = identity_block(layer, (256, 256, 1024))
    layer = identity_block(layer, (256, 256, 1024))
    layer = identity_block(layer, (256, 256, 1024))
    layer = projection_block(layer, (512, 512, 2048))
    layer = identity_block(layer, (512, 512, 2048))
    layer = identity_block(layer, (512, 512, 2048))
    layer = K.layers.AveragePooling2D((7, 7), padding='same',
                                      strides=(1, 1))(layer)
    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer='he_normal')(layer)

    model = K.models.Model(inputs=input_lay, outputs=layer)
    return model
