#!/usr/bin/env python3
""" Exponential """


class Exponential(object):
    """ Exponential class """

    def __init__(self, data=None, lambtha=1.):
        """ Initialization method """
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
                self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period """
        if x < 0:
            return 0
        else:
            return self.lambtha * (2.7182818285 ** -(self.lambtha * x))

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period """
        if x < 0:
            return 0
        else:
            return 1 - (2.7182818285 ** (-self.lambtha * x))
