#!/usr/bin/env python3
""" The Backward Algorithm """
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
        Returns:
            P, B, or None, None on failure.
            P: (float) likelihood of the observations given the model.
            B: (numpy.ndarray) containing the backward path probabilities.
    """
    if type(Observation) != np.ndarray or len(Observation.shape) != 1:
        return None
    if type(Emission) != np.ndarray or len(Emission.shape) != 2:
        return None
    if type(Transition) != np.ndarray or len(Transition.shape) != 2:
        return None
    if type(Initial) != np.ndarray or len(Initial.shape) != 2:
        return None
    T = Observation.shape[0]
    N, M = Emission.shape
    if Transition.shape[0] != N or Transition.shape[0] != Transition.shape[1]:
        return None
    if Initial.shape[0] != N or Initial.shape[1] != 1:
        return None
    B = np.zeros((N, T))
    for i in range(N):
        B[i, T - 1] = 1
    for t in list(range(T - 1))[::-1]:
        for j in range(N):
            B[j, t] = np.sum(Transition[j, :] *
                             Emission[:, Observation[t + 1]] *
                             B[:, t + 1])
    P = np.sum(Initial * Emission[:, Observation[0]] * B[:, 0])

    return P, B
