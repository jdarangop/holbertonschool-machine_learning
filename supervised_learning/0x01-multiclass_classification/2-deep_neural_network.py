#!/usr/bin/env python3
""" Class DeepNeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

    def evaluate(self, X, Y):
        """ Method to evaluate the Neural Network """
        A, cache = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Method to compute the gradient descent """
        m = Y.shape[1]
        for i in range(1, self.L + 1)[::-1]:
            A = cache["A{}".format(i)]
            if i == self.L:
                dZ = A - Y
            else:
                dZ = np.matmul(W.T, dZ) * A * (1 - A)
            dW = (1 / m) * np.matmul(dZ, cache["A" + str(i - 1)].T)
            db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            W = self.weights["W" + str(i)] - (alpha * dW)
            b = self.weights["b" + str(i)] - (alpha * db)
            self.weights["W" + str(i)] = W
            self.weights["b" + str(i)] = b

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

        if type(step) != int:
            raise TypeError('step must be an integer')
        else:
            if step < 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')

        cost_list = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            cost_list.append(self.cost(Y, A))
            if verbose:
                print("Cost after {} iterations: {}"
                      .format(i, self.cost(Y, A)))
        A, cost = self.evaluate(X, Y)
        if verbose:
            print("Cost after {} iterations: {}".format(i + 1, cost))

        if graph:
            plt.plot(list(range(iterations)), cost_list)
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')

        return A, cost

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        fp = open(filename, 'wb')
        pickle.dump(self, fp)
        fp.close()

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            fp = open(filename, 'rb')
            obj = pickle.load(fp)
            fp.close()
            return obj
        except FileNotFoundError:
            return None
