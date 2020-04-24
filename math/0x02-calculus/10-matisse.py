#!/usr/bin/env python3
""" 10 matisse """


def poly_derivative(poly):
    """ find the derivative of  a polynomial
        poly: list with the plynomial
        Return: list with the polynomial of the derivative
    """
    result = []

    if poly is None or type(poly) != list or poly == []:
        return None

    for i in range(len(poly)):
        if type(poly[i]) not in (int, float):
            return None
        elif len(poly) == 1:
            result.append(0)
        else:
            if i == 0:
                continue
            result.append(i * poly[i])

    return result
