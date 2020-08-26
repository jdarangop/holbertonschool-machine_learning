#!/usr/bin/env python3
""" Sparse Autoencoder """
import tensorflow.keras as keras


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """ creates a sparse autoencoder.
        Args:
            input_dims: (int) containing the dimensions of the model input.
            hidden_layers: (list) containing the number of nodes
                           for each hidden layer in the encoder.
            latent_dims: (int) containing the dimensions of the
                         latent space representation.
            lambtha: (float) the regularization parameter used for L1
                     regularization on the encoded output.
        Returns: encoder, decoder, auto
            encoder: (tf.keras.Model) the encoder model.
            decoder: (tf.keras.Model) the decoder model.
            auto: (tf.keras.Model) the full autoencoder model.
    """
    sparse_reg = keras.regularizers.l1(lambtha)
    encoder_input = keras.layers.Input(shape=(input_dims,))
    prev = encoder_input
    for i in hidden_layers:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp
    bottleneck = keras.layers.Dense(latent_dims, activation='relu',
                                    activity_regularizer=sparse_reg)(prev)
    encoder = keras.models.Model(encoder_input, bottleneck)

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    prev = decoder_input
    for i in hidden_layers[::-1]:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(prev)
    decoder = keras.models.Model(decoder_input, output_layer)

    input_layer = keras.layers.Input(shape=(input_dims,))
    encoder_out = encoder(input_layer)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(input_layer, decoder_out)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
