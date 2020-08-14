#!/usr/bin/env python3
""" The Viretbi Algorithm """
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """ calculates the most likely sequence of hidden
        states for a hidden markov model.
        Args:
            Observation: (numpy.ndarray) contains the index of the observation.
            Emission: (numpy.ndarray) containing the emission probability of a
                      specific observation given a hidden state.
            Transition: (numpy.ndarray) containing the transition probabilities
            Initial: (numpy.ndarray) containing the probability of starting in
                     a particular hidden state.
        Returns:
            path, P, or None, None on failure.
            path: (list) containing the most likely sequence of hidden states.
            P: (float) the probability of obtaining the path sequence.
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
    viterbi = np.zeros((N, T))
    backp = np.zeros((N, T))
    for i in range(N):
        viterbi[i, 0] = Initial[i, 0] * Emission[i, Observation[0]]
        backp[i, 0] = 0
    for t in range(1, T):
        for j in range(N):
            viterbi[j, t] = np.amax(Transition[:, j] * viterbi[:, t - 1] *
                                    Emission[j, Observation[t]])
            backp[j, t] = np.argmax(Transition[:, j] * viterbi[:, t - 1] *
                                    Emission[j, Observation[t]])
    P = np.amax(viterbi[:, T - 1])
    tmp = np.argmax(viterbi[:, T - 1])
    path = [tmp, tmp]
    for i in list(range(1, T - 1))[::-1]:
        tmp = int(backp[tmp, i])
        path = [tmp] + path

    return path, P
