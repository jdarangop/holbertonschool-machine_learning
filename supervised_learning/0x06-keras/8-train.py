#!/usr/bin/env python3
""" Train """
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
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
            learning_rate_decay: (bool) indicates wheter learning rate decay
                                 should be used.
            alpha: (float) initial learning rate.
            decay_rate: (float) the decay rate.
            save_best: (bool) indicating whether to save the model
                       after each epoch.
            filepath: (str) path where the model should be saved.
        Returns:
            (tf.History) object generated after training the model.
    """
    def learning_rate_decay(epoch):
        """ learning rate callback """
        alpha_utd = alpha / (1 + (decay_rate * epoch))
        return alpha_utd

    callbacks = []
    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(filepath=filepath,
                                                 save_best_only=True,
                                                 monitor='val_loss',
                                                 mode='min')
        callbacks.append(checkpoint)
    if validation_data and learning_rate_decay:
        decay = K.callbacks.LearningRateScheduler(learning_rate_decay,
                                                  verbose=1)
        callbacks.append(decay)
    if validation_data and early_stopping:
        EarlyStopping = K.callbacks.EarlyStopping(patience=patience,
                                                  monitor='val_loss',
                                                  mode='min')
        callbacks.append(EarlyStopping)

    return network.fit(data, labels, batch_size=batch_size,
                       epochs=epochs, verbose=verbose,
                       shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
