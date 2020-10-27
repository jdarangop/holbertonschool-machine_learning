#!/usr/bin/env python3
""" From Numpy """
import pandas as pd


def from_numpy(array):
    """ creates a pd.DataFrame from a np.ndarray.
        Args:
            array: (np.ndarray)  from which you should
                   create the pd.DataFrame.
        Returns:
            (pd.DataFrame) Dataframe.
    """
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    return pd.DataFrame(array,
                        columns=[alphabet[i]
                                 for i in range(array.shape[1])])
