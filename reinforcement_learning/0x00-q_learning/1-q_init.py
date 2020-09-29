#!/usr/bin/env python3
""" Initialize Q-table """
import numpy as np


def q_init(env):
    """ initializes the Q-table.
        Args:
            env: FrozenLakeEnv instance.
        Returns:
            the Q-table as a numpy.ndarray of zeros.
    """
    return np.zeros((env.observation_space.n, env.action_space.n))
