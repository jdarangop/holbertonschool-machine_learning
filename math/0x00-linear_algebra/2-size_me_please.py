#!/usr/bin/env python3
""" File Size Me Please """


def matrix_shape(matrix):
    """ find the shape of a matrix
        matrix: list
    """
    dim = []
    var = matrix.copy()
    while(type(var) == list):
        dim.append(len(var))
        var = var[0]

    return dim
