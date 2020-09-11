#!/usr/bin/env python3
""" Forecast BTC """
import numpy as np
import tensorflow as tf


def load_data(input_width, label_width, shift, train_df, val_df, test_df):
    """ load the data to a tf.data.Dataset
        Args:
            input_width: (int) input width of the window.
            label_width: (int) label width of the window.
            shift: (int) shift of the window.
            train_df: (numpy.ndarray) train array.
            val_df: (numpy.ndarray) validation array.
            test_df: (numpy.ndarray) test array.
        Returns:
            (tf.data.Dataset) Train values.
            (tf.data.Dataset) Validation values.
            (tf.data.Dataset) Test values.
    """
    total_size = input_width + shift
    input_slice = slice(0, input_width)
    input_indices = np.arange(total_size)[input_slice]

    label_start = total_size - label_width
    labels_slice = slice(label_start, None)
    label_indices = np.arange(total_size)[labels_slice]

    def split(features):
        """ split the data in windows.
            Args:
                features: array to split.
            Return:
                (tuple) with inputs and labels.
        """
        inputs = features[:, input_slice, :]
        labels = features[:, labels_slice, :]

        inputs.set_shape([None, input_width, None])
        labels.set_shape([None, label_width, None])

        return inputs, labels

    def make_dataset(data):
        """ convert the array to a tf.data.Dataset type.
            Args:
                data: (numpy.ndarray) data to convert.
            Returns:
                (tf.data.Dataset) data converted.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)
        # ds = tf.data.Dataset.from_tensor_slices(data)

        # ds.window(24)
        ds = ds.map(split)

        return ds

    train = make_dataset(train_df)
    val = make_dataset(val_df)
    test = make_dataset(test_df)

    return train, val, test


def compile_and_fit(model, train_res, val_res, patience=2):
    """ compile and fit the model.
        Args:
            model: (tf.Model) model to be fitted.
            train_res: (tf.data.Dataset) train values.
            val_res: (tf.data.Dataset) validation values.
            patience: (int) patience of the model.
        Returns:
            (tf.Model) model compiled.
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.mean_squared_error(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.metrics.mean_squared_error()])

    history = model.fit(train_res, epochs=20,
                        validation_data=val_res,
                        callbacks=[early_stopping])

    return history


def model():
    """ build the model.
        Args:
            None.
        Returns:
            (tf.Model) the RNN model.
    """
    lstm_model = tf.keras.models.Sequential([
                 tf.keras.layers.LSTM(32,
                                      input_shape=[24, 7],
                                      return_sequences=True),
                 tf.keras.layers.Dense(units=1)
                                            ])

    return lstm_model
