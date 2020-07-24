#!/usr/bin/env python3
""" Determinant """


def recursion_determinant(mat):
    """ operate the recursion in the determinant.
        Args:
            mat: (list/int) temporal matrix to calculate.
        Returns:
            (list/int) the result.
    """
    if len(mat) == 1:
        return mat[0][0]
    else:
        result = 0
        for i in range(len(mat)):
            tmp = [j.copy() for j in mat]
            indexs = tmp.pop(0)
            for k in tmp:
                k.pop(i)
            result += ((-1) ** i) * indexs[i] * recursion_determinant(tmp)

        return result


def determinant(matrix):
    """ calculate the determinant of a matrix.
        Args:
            matrix: (list)  list of lists which containg a matrix.
        Returns:
            (float) the determinant of the matrix.
    """
    if type(matrix) != list or matrix == [] or type(matrix[0]) != list:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        return 1
    num_rows = len(matrix)
    for i in matrix:
        if type(i) != list:
            raise TypeError('matrix must be a list of lists')
        if len(i) != num_rows:
            raise ValueError('matrix must be a square matrix')
    return recursion_determinant(matrix)
