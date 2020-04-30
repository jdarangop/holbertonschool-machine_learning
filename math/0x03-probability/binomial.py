#!/usr/bin/env python3
""" Binomial """


class Binomial(object):
    """ Binomial class """

    def __init__(self, data=None, n=1, p=0.5):
        """ Initi method """
        if data is None:
            if n < 0:
                raise ValueError('n must be a positive value')
            elif p < 0 or p > 1:
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
                var = sum([(i - mean) ** 2 for i in data]) / len(data)
                pro = 1 - (var/mean)
                n = mean / pro
                self.n = int(round(n))
                self.p = float(mean / self.n)

    @staticmethod
    def factorial(n):
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes """
        result = ((Binomial.factorial(self.n) /
                  (Binomial.factorial(k) * Binomial.factorial(self.n - k))) *
                  ((self.p ** k) *
                   ((1 - self.p) ** (self.n - k))))
        return result

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes """
        result = 0
        for i in range(k + 1):
            result += self.pmf(i)
        return result
