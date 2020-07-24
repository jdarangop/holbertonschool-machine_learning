#!/usr/bin/env python3
""" Minor """


def determinant(mat):
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
            result += ((-1) ** i) * indexs[i] * determinant(tmp)

        return result


def minor(matrix):
    """ calculates the minor matrix of a matrix.
        Args:
            matrix: (list) list of lists whose minor
                    matrix should be calculated.
        Returns:
            (list) the minor matrix of matrix
    """
    if type(matrix) != list or matrix == [] or type(matrix[0]) != list:
        raise TypeError('matrix must be a list of lists')
    if matrix == [[]]:
        raise ValueError('matrix must be a non-empty square matrix')
    num_rows = len(matrix)
    if num_rows == 1:
        return [[1]]
    for i in matrix:
        if type(i) != list:
            raise TypeError('matrix must be a list of lists')
        if len(i) != num_rows:
            raise ValueError('matrix must be a non-empty square matrix')
    result = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            tmp = [w.copy() for w in matrix]
            tmp.pop(i)
            for k in tmp:
                k.pop(j)
            row.append(determinant(tmp))
        result.append(row)

    return result
