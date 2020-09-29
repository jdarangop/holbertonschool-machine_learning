#!/usr/bin/env python3
""" Epsilon Greedy """
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """ uses epsilon-greedy to determine the next action.
        Args:
            Q: (numpy.ndarray) containing the q-table.
            state: the current state.
            epsilon: the epsilon to use for the calculation.
        Returns:
            the next action index.
    """
    if epsilon > np.random.uniform(0, 1):
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state, :])
