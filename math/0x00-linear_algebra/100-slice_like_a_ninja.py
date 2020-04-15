#!/usr/bin/env python3
""" Slice Like A Ninja """


def np_slice(matrix, axes={}):
    """ slices a matrxi along a specific axes
        matrix: numpy ndarray
        axes: dict
    """
    list_slices = []
    for i in range(matrix.ndim):
        tupla = axes.get(i)
        list_slices.append(slice(None) if tupla is None else slice(*tupla))
    tuple_slices = tuple(list_slices)
    return matrix[tuple_slices]
