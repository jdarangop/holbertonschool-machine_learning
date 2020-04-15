#!/usr/bin/env python3
""" Line Up """


def add_arrays(arr1, arr2):
    """ add two arrays
        arr1: list
        arr2: list
    """
    if len(arr1) != len(arr2):
        return None
    else:
        result_add = []
        for i in range(len(arr1)):
            result_add.append(arr1[i] + arr2[i])

    return result_add
