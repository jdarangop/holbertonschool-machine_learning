#!/usr/bin/env python3
""" Convolutional Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ creates a convolutional autoencoder.
        Args:
            input_dims: (tuple) containing the dimensions of the model input.
            filters: (list) containing the number of filters for
                     each convolutional layer in the encoder.
            latent_dims: (tuple) containing the dimensions of the
                         latent space representation.
        Returns: encoder, decoder, auto
            encoder: (tf.keras.Model) the encoder model.
            decoder: (tf.keras.Model) the decoder model.
            auto: (tf.keras.Model) the full autoencoder model.
    """
    encoder_input = keras.layers.Input(shape=input_dims)
    prev = encoder_input
    for i in filters:
        conv = keras.layers.Conv2D(i, kernel_size=(3, 3),
                                   padding='same', activation='relu')(prev)
        pool = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         padding='same')(conv)
        prev = pool
    encoder = keras.models.Model(encoder_input, prev)

    decoder_input = keras.layers.Input(shape=latent_dims)
    prev = decoder_input
    reverse = filters[::-1]
    for i in range(len(reverse)):
        if i == len(reverse) - 1:
            conv = keras.layers.Conv2D(reverse[i], kernel_size=(3, 3),
                                       padding='valid',
                                       activation='relu')(prev)
        else:
            conv = keras.layers.Conv2D(reverse[i], kernel_size=(3, 3),
                                       padding='same', activation='relu')(prev)
        upsamp = keras.layers.UpSampling2D(size=(2, 2))(conv)
        prev = upsamp
    output_layer = keras.layers.Conv2D(input_dims[2], kernel_size=(3, 3),
                                       padding='same',
                                       activation='sigmoid')(prev)
    decoder = keras.models.Model(decoder_input, output_layer)

    input_layer = keras.layers.Input(shape=input_dims)
    encoder_out = encoder(input_layer)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(input_layer, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
