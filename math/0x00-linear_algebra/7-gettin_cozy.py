#!/usr/bin/env python3
""" Gettin Cozy """


def cat_matrices2D(mat1, mat2, axis=0):
    """ concatenate two 2D matrices acord to the axis
        mat1: list of lists
        mat2: list of lists
        axis: int
    """
    result_mat = [i.copy() for i in mat1]
    mat2_copy = [i.copy() for i in mat2]
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            for i in mat2:
                result_mat.append(i)
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            for i in range(len(mat1)):
                result_mat[i] += mat2_copy[i]

    return result_mat
