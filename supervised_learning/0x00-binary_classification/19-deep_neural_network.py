#!/usr/bin/env python3
""" Class DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork(object):
    """ DeepNeuralNetwork """

    def __init__(self, nx, layers):
        """ Init method """
        if type(nx) != int:
            raise TypeError('nx must be an integer')
        else:
            if nx < 1:
                raise ValueError('nx must be a positive integer')

        if type(layers) != list:
            raise TypeError('layers must be a list of positive integers')
        else:
            if any(list(map(lambda x: x <= 0, layers))):
                raise TypeError('layers must be a list of positive integers')
            if len(layers) < 1:
                raise TypeError('layers must be a list of positive integers')
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            key = "W{}".format(i + 1)
            if i == 0:
                self.weights[key] = np.random.randn(layers[i],
                                                    nx)*np.sqrt(2/nx)
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            else:
                self.weights[key] = (np.random.randn(layers[i],
                                                     layers[i - 1]) *
                                     np.sqrt(2/layers[i - 1]))
                self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """ Method fo forward propagation """
        self.__cache['A0'] = X
        for i in range(self.L):
            keyA = "A{}".format(i + 1)
            keyb = "b{}".format(i + 1)
            keyAo = "A{}".format(i)
            keyW = "W{}".format(i + 1)
            self.__cache[keyA] = 1.0 / (1.0 +
                                        np.exp(-(np.matmul(self.weights[keyW],
                                                           self.cache[keyAo])
                                               + self.weights[keyb])))

        return self.cache[keyA], self.cache

    def cost(self, Y, A):
        """ Method to compute the Cost """
        m = Y.shape[1]
        cost = np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost
