#!/usr/bin/env python3
""" Play """
import numpy as np


def play(env, Q, max_steps=100):
    """ the trained agent play an episode.
        Args:
            env: the FrozenLakeEnv instance.
            Q: (numpy.ndarray) containing the Q-table.
            max_steps: the maximum number of steps in the episode.
        Returns:
            the total rewards for the episode.
    """
    state = env.reset()
    env.render()

    for step in range(max_steps):
        action = np.argmax(Q[state, :])

        state, reward, done, info = env.step(action)

        env.render()

        if done is True:
            break

    return reward
