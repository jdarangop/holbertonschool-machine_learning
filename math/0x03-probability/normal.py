#!/usr/bin/env python3
""" Normal Distribution """


class Normal(object):
    """ Normal class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """ Init method """
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if type(data) != list:
                raise TypeError('data must be a list')
            elif len(data) <= 2:
                raise ValueError('data must contain multiple values')
            else:
                self.mean = float(sum(data) / len(data))
                self.stddev = float((sum([(i - self.mean) ** 2
                                          for i in data])
                                    / len(data)) ** (1/2))

    def z_score(self, x):
        """ Calculates the z-score of a given x-value """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score """
        return self.mean + (self.stddev * z)

    def pdf(self, x):
        """ Calculates the value of the PDF for a given x-value """
        result = ((1/(self.stddev * ((2 * 3.1415926536) ** (1/2))))
                  * (2.7182818285 ** (-((x - self.mean) ** 2) /
                                      (2 * (self.stddev ** 2)))))
        return result

    def cdf(self, x):
        """ Calculates the value of the CDF for a given x-value """
        z = (x - self.mean) / (self.stddev * (2 ** (1/2)))
        result = (0.5 * (1 + (2 / (3.1415926536 ** (1/2)))
                  * (z - ((z**3) / 3) + ((z**5) / 10)
                     - ((z**7) / 42) + ((z**9) / 216))))
        return result
