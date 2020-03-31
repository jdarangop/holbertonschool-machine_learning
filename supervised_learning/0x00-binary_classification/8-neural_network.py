#!/usr/bin/env python3
""" Class Neural Network """
import numpy as np


class NeuralNetwork(object):
    """Neural Network"""
    def __init__(self, nx, nodes):
        """ Init Method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        else:
            if nx < 1:
                raise ValueError('nx must be a positive integer')

        if type(nodes) != int:
            raise TypeError('nodes must be an integer')
        else:
            if nodes < 1:
                raise ValueError('nodes must be a positive integer')

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
