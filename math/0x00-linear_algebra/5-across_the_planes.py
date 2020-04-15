#!/usr/bin/env python3
""" Across The Planes """


def add_matrices2D(mat1, mat2):
    """ compute the add 2D matrices
        mat1: list of lists
        mat2: list of lists
    """
    if len(mat1) != len(mat2):
        return None
    else:
        result_add = []
        for i in range(len(mat1)):
            if len(mat1[i]) != len(mat2[i]):
                return None
            else:
                temp_row = []
                for j in range(len(mat1[i])):
                    temp_row.append(mat1[i][j] + mat2[i][j])

            result_add.append(temp_row)

    return result_add
