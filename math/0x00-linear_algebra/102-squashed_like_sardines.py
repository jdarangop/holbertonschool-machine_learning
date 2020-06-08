#!/usr/bin/env python3
""" Squashed Like Sardines """


def dim(matrix):
    """ find the dimention of a array
        matrix: list
    """

    if type(matrix[0]) in (int, float):
        return 0
    else:
        return 1 + dim(matrix[0])


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


def concatenate(mat1, mat2, axis=0):
    """ concatenate process
    """

    if dim(mat1) != dim(mat2) or axis > dim(mat1) or axis > dim(mat1):
        return None
    else:
        if axis == 0:
            result = [*mat1, *mat2]
            return result
        else:
            result = []
            for i in range(len(mat1)):
                result.append(cat_matrices(mat1[i], mat2[i], axis - 1))
            if None in result:
                return None
            else:
                return result


def cat_matrices(mat1, mat2, axis=0):
    """ concatenate two matrices.
        Args:
            mat1: (list) matrix one.
            mat2: (list) matrix two.
            axis: (int) axis to concatenate.
        Returns:
            (list) with the matrix concatenate.
            (None) if it's no posible.
    """
    array1 = matrix_shape(mat1)
    array2 = matrix_shape(mat2)
    if axis < len(array1) and axis < len(array2):
        array1.pop(axis)
        array2.pop(axis)
    else:
        return None

    if array1 != array2:
        return None
    else:
        return concatenate(mat1, mat2, axis)
