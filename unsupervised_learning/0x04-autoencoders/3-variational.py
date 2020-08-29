#!/usr/bin/env python3
""" Vanilla Autoencoder """
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """ creates a variational autoencoder.
        Args:
            input_dims: (int) containing the dimensions of the model input.
            hidden_layers: (list) containing the number of nodes
                           for each hidden layer in the encoder.
            latent_dims: (int) containing the dimensions of the
                         latent space representation.
        Returns: encoder, decoder, auto
            encoder: (tf.keras.Model) the encoder model.
            decoder: (tf.keras.Model) the decoder model.
            auto: (tf.keras.Model) the full autoencoder model.
    """
    encoder_input = keras.layers.Input(shape=(input_dims,))
    prev = encoder_input
    for i in hidden_layers:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp
    mean = keras.layers.Dense(latent_dims, activation=None)(prev)
    sigma = keras.layers.Dense(latent_dims, activation=None)(prev)

    def sampling(args):
        """ sampling function. """
        mean, sigma = args
        batch_size = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch_size, dim))
        return mean + keras.backend.exp(sigma) * epsilon
    bottleneck = keras.layers.Lambda(sampling,
                                     output_shape=(latent_dims,))([mean,
                                                                   sigma])
    encoder = keras.models.Model(encoder_input, [bottleneck, mean, sigma])

    decoder_input = keras.layers.Input(shape=(latent_dims,))
    prev = decoder_input
    for i in hidden_layers[::-1]:
        tmp = keras.layers.Dense(i, activation='relu')(prev)
        prev = tmp
    output_layer = keras.layers.Dense(input_dims, activation='sigmoid')(prev)
    decoder = keras.models.Model(decoder_input, output_layer)

    # input_layer = keras.layers.Input(shape=(input_dims,))
    encoder_out, m, sig = encoder(encoder_input)
    decoder_out = decoder(encoder_out)
    auto = keras.models.Model(encoder_input, decoder_out)

    def va_loss(y_actual, y_predicted):
        """ Variational Autoencoder loss function. """
        loss = keras.backend.binary_crossentropy(y_actual, y_predicted)
        loss_sum = keras.backend.sum(loss, axis=1)
        kl_divergence = (-0.5 * keras.backend.mean(1 + sigma -
                                                   keras.backend.square(mean) -
                                                   keras.backend.exp(sigma),
                                                   axis=-1))
        return loss_sum + kl_divergence

    # auto.compile(optimizer='adam', loss='binary_crossentropy')
    auto.compile(optimizer='adam', loss=va_loss)

    return encoder, decoder, auto
