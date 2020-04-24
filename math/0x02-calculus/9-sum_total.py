#!/usr/bin/env python3
""" 9 sum_total """


def summation_i_squared(n):
    """ compute the summatory of the num to square from 1.
        n: is the stopping condition.
    """
    if n < 1:
        return None
    else:
        return sum(map(lambda x: x**2, range(1, n + 1)))
