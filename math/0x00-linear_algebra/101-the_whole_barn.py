#!/usr/bin/env python3
""" The Whole Barn """


def add_matrices(mat1, mat2):
    """ compute the between to matrices
        mat1: list / could have more than one nested list
        mat2: list / could have more than one nested list
    """
    if len(mat1) != len(mat2):
        return None
    else:
        if type(mat1[0]) in (int, float):
            result = []
            for i in range(len(mat1)):
                result.append(mat1[i] + mat2[i])
            return result
        else:
            result = []
            for i in range(len(mat1)):
                result.append(add_matrices(mat1[i], mat2[i]))
            if None in result:
                return None
            else:
                return result
