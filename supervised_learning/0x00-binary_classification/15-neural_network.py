#!/usr/bin/env python3
""" Class Neural Network """
import numpy as np
import matplotlib.pyplot as plt


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

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """ Method for Forward Propagation """
        self.__A1 = 1.0 / (1.0 + np.exp(-(np.matmul(self.W1, X) + self.b1)))
        self.__A2 = 1.0 / (1.0 + np.exp(-(np.matmul(self.W2, self.__A1)
                           + self.b2)))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Method to compute the Cost """
        m = Y.shape[1]
        cost = np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Method to evaluate the Neural Network """
        A1, A2 = self.forward_prop(X)
        return np.where(self.__A2 >= 0.5, 1, 0), self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Method to compute the gradient descent """
        m = Y.shape[1]
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.matmul(A1, dZ2.T).T
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.matmul(self.W2.T, dZ2) * A1 * (1 - A1)
        dW1 = (1 / m) * np.matmul(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__b1 = self.__b1 - (alpha * db1)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
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

        if graph is True or verbose is True:
            if type(step) != int:
                raise TypeError('step must be an integer')
            else:
                if step < 0 or step > iterations:
                    raise ValueError('step must be positive and <= iterations')

        cost_list = []

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
            cost_list.append(self.cost(Y, A2))
            if i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A2)))
        A, cost = self.evaluate(X, Y)
        if verbose:
            print("Cost after {} iterations: {}".format(i + 1, cost))

        if graph:
            plt.plot(list(range(iterations)), cost_list)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')

        return self.evaluate(X, Y)
