#!/usr/bin/env python3
""" Ridin Bareback """


def mat_mul(mat1, mat2):
    """ computes the multiplication between two matrices
        mat1: list of lists
        mat2: list of lists
    """
    mat1_copy = [i.copy() for i in mat1]
    mat2_copy = [i.copy() for i in mat2]
    if len(mat1[0]) != len(mat2):
        return None
    else:
        result = [[0 for j in range(len(mat2[0]))] for i in range(len(mat1))]
        for i in range(len(mat1)):
            for j in range(len(mat2[0])):
                for z in range(len(mat2)):
                    result[i][j] += mat1_copy[i][z] * mat2_copy[z][j]

    return result
