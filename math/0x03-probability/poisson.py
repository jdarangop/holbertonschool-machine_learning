#!/usr/bin/env python3
""" Poisson """


class Poisson(object):
    """ Poisson class """

    def __init__(self, data=None, lambtha=1.):
        """ Initicialization of the object """
        if data is None:
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            else:
                self.lambtha = float(lambtha)
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                self.lambtha = float(sum(data) / len(data))

    @staticmethod
    def factorial(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes """
        if k <= 0:
            return 0
        factorial = 1
        for i in range(1, int(k) + 1):
            factorial *= i
        result = ((2.7182818285 ** -self.lambtha) *
                  (self.lambtha ** int(k)) / factorial)
        return result

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes """
        if k < 0:
            return 0
        result = 0
        for i in range(int(k) + 1):
            result += ((self.lambtha ** i) / Poisson.factorial(i))
        return result * (2.7182818285 ** -self.lambtha)
