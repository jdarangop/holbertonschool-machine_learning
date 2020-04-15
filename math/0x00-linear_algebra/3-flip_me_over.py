#!/usr/bin/env python3
""" Flip Me Over """


def matrix_transpose(matrix):
    """ compute the transpose of a matrix
        matrix: list
    """
    matrix_T = []
    for i in range(len(matrix[0])):
        column = []
        for j in range(len(matrix)):
            column.append(matrix[j][i])
        matrix_T.append(column)

    return matrix_T
