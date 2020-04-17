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


def cat_matrices(mat1, mat2, axis=0):
    """ concatenate two matrices
        mat1: list
        mat2: list
        axis: int
    """

    if axis == 0:
        if dim(mat1) != dim(mat2):
            return None
        else:
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
