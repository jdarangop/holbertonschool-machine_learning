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
        self.__A = self.__A = 1.0/(1.0 +
                                   np.exp(-(np.matmul(self.W, X)
                                          + self.b)))
        return self.__A

    def cost(self, Y, A):
        """ Method to compute the cost """
        m = Y.shape[1]
        cost = np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Method to evaluate the neuron """
        Z = self.forward_prop(X)
        return np.where(self.__A >= 0.5, 1, 0), self.cost(Y, Z)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Method to compute the gradient descent """
        m = Y.shape[1]
        dZ = A - Y
        dW = (1 / m) * np.matmul(X, dZ.T).T
        db = (1 / m) * np.sum(dZ)
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Method to train the neuron """
        if type(iterations) != int:
            raise TypeError('iterations must be an integer')
        else:
            if iterations < 0:
                raise ValueError('iterations must be a positive integer')

        if type(alpha) != float:
            raise TypeError('alpha must be a float')
        else:
            if alpha < 0:
                raise ValueError('alpha must be positive')

        for i in range(iterations):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)

        return self.evaluate(X, Y)
