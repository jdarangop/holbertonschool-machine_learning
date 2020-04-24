#!/usr/bin/env python3
""" Integrate """


def poly_integral(poly, C=0):
    """ Find the integrate of a polynomial
        poly: list with the coeficients of the polinomial
        C: int constant of integration
        Return: list with the coeficients after the integration
    """
    if C is None or type(C) not in (int, float):
        return None

    result = [C]
    for i in range(len(poly)):
        if poly[i] is None or type(poly[i]) not in (int, float):
            return None
        result.append(poly[i] / (i + 1))

    return result