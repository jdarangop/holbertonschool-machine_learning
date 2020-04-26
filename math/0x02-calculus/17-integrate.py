#!/usr/bin/env python3
""" Integrate """


def poly_integral(poly, C=0):
    """ Find the integrate of a polynomial
        poly: list with the coeficients of the polinomial
        C: int constant of integration
        Return: list with the coeficients after the integration
    """
    # if C is None or type(C) not in (int, float):
    if C is None or type(C) != int:
        return None

    if poly is None or type(poly) != list or poly == []:
        return None

    result = [C]
    for i in range(len(poly)):
        if poly[i] is None or type(poly[i]) not in (int, float):
            return None
        coeficient = poly[i] / (i + 1)
        if poly[i] % (i + 1) == 0:
            coeficient = int(coeficient)
        result.append(coeficient)

    return result
