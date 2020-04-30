#!/usr/bin/env python3
""" Binomial """
import math


class Binomial(object):
    """ Binomial class """

    def __init__(self, data=None, n=1, p=0.5):
        """ Initi method """
        if data is None:
            if n < 0:
                raise ValueError('n must be a positive value')
            elif p < 0 and p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            else:
                self.n = int(n)
                self.p = float(p)
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            elif len(data) < 2:
                raise ValueError('data must contain multiple values')
            else:
                mean = sum(data) / int(len(data))
                pro = 2 * (mean / int(len(data)))
                self.n = int(mean / pro)
                self.p = float(mean / self.n)

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes """
        result = ((math.factorial(self.n) /
                  (math.factorial(k) * math.factorial(self.n - k))) *
                  ((self.p ** k) *
                   ((1 - self.p) ** (self.n - k))))
        return result

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes """
        result = 0
        for i in range(k + 1):
            result += self.pmf(i)
        return result
