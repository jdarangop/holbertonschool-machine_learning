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

    def forward_prop(self, X):
        """ Method for Forward Propagation """
        self.__A = self.__A = 1.0/(1.0 + np.exp(-(np.dot(self.W, X) + self.b)))
        return self.__A

    def cost(self, Y, A):
        """ Method to compute the cost """
        #h = Neuron.sigmoid(A)
        """cost_log = np.dot(h, np.log(Y.T + 0.0000001))
        cost_log += np.dot((1.0000001 - h), np.log(1.0000001 - Y.T))
        cost_log *= (-1/Y.shape[1])
        cost = (-1/Y.shape[1]) * np.sum( np.multiply(np.log(A), Y) + np.multiply(np.log(1.0000001 - A), (1.0000001 - Y)))
        error = (-Y * np.log(A)) - ((1.0000001 - Y) * np.log(1.0000001 - A))
        cost = (1/Y.shape[1]) * sum(error)"""
        cost = (-1.0) * np.mean(np.matmul(Y, np.log(A)) + np.matmul((1.0000001 - Y), np.log(1.0000001 - A)), axis=1)
        return cost[0]
