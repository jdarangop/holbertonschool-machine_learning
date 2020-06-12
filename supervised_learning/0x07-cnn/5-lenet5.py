#!/usr/bin/env python3
""" LeNet-5 (Keras) """
import tensorflow.keras as K


def lenet5(x):
    """ builds a modified version of the LeNet-5
        architecture using keras.
        Args:
            X: (K.Input) containing the input images
               for the network.
        Returns:
            -(K.Model) compiled to use Adam optimization.
            -accuracy
    """
    # model = K.Sequential()
    conv_lay1 = K.layers.Conv2D(filters=6,
                                kernel_size=(5, 5),
                                padding='same',
                                activation='relu',
                                kernel_initializer='he_normal')(x)
    pool_lay1 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                      strides=(2, 2))(conv_lay1)
    conv_lay2 = K.layers.Conv2D(filters=16,
                                kernel_size=(5, 5),
                                padding='valid',
                                activation='relu',
                                kernel_initializer='he_normal')(pool_lay1)
    pool_lay2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                      strides=(2, 2))(conv_lay2)
    flatten = K.layers.Flatten(input_shape=(28, 28))(pool_lay2)
    full_lay3 = K.layers.Dense(units=120,
                               activation='relu',
                               kernel_initializer='he_normal')(flatten)
    full_lay4 = K.layers.Dense(units=84,
                               activation='relu',
                               kernel_initializer='he_normal')(full_lay3)
    softmax = K.layers.Dense(units=10,
                             activation='softmax',
                             kernel_initializer='he_normal')(full_lay4)
    model = K.Model(inputs=x, outputs=softmax)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
