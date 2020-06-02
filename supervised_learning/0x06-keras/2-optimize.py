#!/usr/bin/env python3
""" Optimize """
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """ sets up Adam optimization for a keras model
        with categorical crossentropy and accuracy metrics.
        Args:
            network: (tf.Tensor) the model to optimize.
            alpha: (float) learning rate.
            beta1: (float) first Adam optimization parameter.
            beta2: (float) second Adam optimization parameter.
        Returns:
            None.
    """
    Adam = K.optimizers.Adam(beta_1=beta1, beta_2=beta2)
    network.compile(optimizer=Adam,
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])
