#!/usr/bin/env python3
""" The Forward Algorithm """
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
        Returns:
            P, F, or None, None on failure.
            P: (float) likelihood of the observations given the model.
            F: (numpy.ndarray) containing the forward path probabilities.
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
    F = np.zeros((N, T))
    for i in range(N):
        F[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]
    # F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(Transition[:, j] * F[:, t - 1] *
                             Emission[j, Observation[t]])
    P = np.sum(F[:, T - 1])

    return P, F
