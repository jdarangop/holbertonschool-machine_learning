#!/usr/bin/env python3
""" Inverse """


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


def cofactor(matrix):
    """ calculates the cofactor matrix of a matrix.
        Args:
            matrix: (list)  a list of lists whose
                    cofactor matrix should be calculated.
        Returns:
            (list) the cofactor matrix of matrix.
    """
    minor_mat = minor(matrix)
    for i in range(len(minor_mat)):
        for j in range(len(minor_mat[0])):
            minor_mat[i][j] *= ((-1)**(i + j))
    return minor_mat


def adjugate(matrix):
    """ calculates the adjugate matrix of a matrix.
        Args:
            matrix: (list)  a list of lists whose
                    adjugate matrix should be calculated.
        Returns:
            (list) the adjugate matrix of matrix.
    """

    cofactor_mat = cofactor(matrix)

    result = []
    for i in range(len(cofactor_mat[0])):
        tmp = []
        for j in range(len(cofactor_mat)):
            tmp.append(0)
        result.append(tmp)

    for i in range(len(cofactor_mat)):
        for j in range(len(cofactor_mat[0])):
            result[j][i] = cofactor_mat[i][j]

    return result


def inverse(matrix):
    """ calculates the inverse matrix of a matrix.
        Args:
            matrix: (list)  a list of lists whose
                    inverse matrix should be calculated.
        Returns:
            (list) the inverse matrix of matrix.
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

    det = determinant(matrix)
    if det == 0:
        return None
    adjugate_mat = adjugate(matrix)
    for i in range(len(adjugate_mat)):
        for j in range(len(adjugate_mat)):
            adjugate_mat[i][j] /= det
    return adjugate_mat
