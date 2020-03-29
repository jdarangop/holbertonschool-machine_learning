#!/usr/bin/env python3
""" Class Neuron """
import numpy as np


class Neuron(object):
    """Neuron"""
    def __init__(self, nx):
        """ Init method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        else:
            if nx < 1:
                raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Getter W """
        return self.__W

    @property
    def b(self):
        """ Getter b """
        return self.__b

    @property
    def A(self):
        """ Getter A """
        return self.__A
