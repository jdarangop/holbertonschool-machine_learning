#!/usr/bin/env python3
""" Preprocess the Data """
import numpy as np


def preprocess_data(path):
    """ preprocess the data.
        Args:
            path: (str) path where the file it's located.
        Returns:
            (numpy.ndarray) containing the train values.
            (numpy.ndarray) containing the validation values.
            (numpy.ndarray) containing the test values.
    """

    np_array = np.genfromtxt(path, delimiter=',')

    np_array_clean = np_bitstamp[np.logical_not(np.any(np.isnan(np_array),
                                                       axis=1))]

    n = len(np_array_clean)

    train_df = np_bitstamp_clean[0: int(n * 0.7)]
    val_df = np_bitstamp_clean[int(n * 0.7): int(n * 0.9)]
    test_df = np_bitstamp_clean[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df
