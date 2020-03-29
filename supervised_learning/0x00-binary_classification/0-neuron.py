#!/usr/bin/env python3
""" Class Neuron """
import numpy as np

class Neuron(object):
    """Neuron"""
    def __init__(self, nx):
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        else:
            if nx < 1:
                raise ValueError('nx must be a positive integer')
        self.W = np.random.randn(1, nx) * 0.01
        self.b = 0
        self.A = 0
