#!/usr/bin/env python3
""" From File """
import pandas as pd


def from_file(filename, delimiter):
    """ loads data from a file as a pd.DataFrame.
        Args:
            filename: (str) the path to the file.
            delimiter: (str) the cloumn separator.
        Returns:
            (pd.DataFrame) the DataFrame loaded.
    """
    return pd.read_table(filename, delimiter=delimiter)
