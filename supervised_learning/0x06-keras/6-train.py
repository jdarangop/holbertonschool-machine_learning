#!/usr/bin/env python3
""" Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ trains a model using mini-batch gradient descent.
        Args:
            network: (tf.Tensor) model to train.
            data: (numpy.ndarray) containing the input data.
            labels: (numpy.ndarray) One-Hot array containing
                    the labels of data.
            batch_size: (int) the size of mini-batch.
            epochs: (int) number of epochs.
            verbose: (bool) determines if output should be
                     printed during training.
            shuffle: (bool) determines whether to shuffle the batches.
            validation_data: (tuple) containing the data to validate the model.
            early_stopping: (bool) incicates if early stopping should be used.
            patience: (int) patience using for early stopping.
        Returns:
            (tf.History) object generated after training the model.
    """
    callbacks = []
    if early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience)
        callbacks.append(EarlyStopping)
    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
