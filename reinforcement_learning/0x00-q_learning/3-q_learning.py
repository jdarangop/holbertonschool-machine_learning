#!/usr/bin/env python3
""" Q-learning """
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1,
          epsilon_decay=0.05):
    """ performs Q-learning.
        Args:
            env: the FrozenLakeEnv instance.
            Q: (numpy.ndarray) containing the Q-table.
            episodes: the total number of episodes to train over.
            max_steps: the maximum number of steps per episode.
            alpha: the learning rate.
            gamma: the discount rate.
            epsilon: the initial threshold for epsilon greedy.
            min_epsilon: the minimum value that epsilon should decay to.
            epsilon_decay: the decay rate for updating
                           epsilon between episodes.
        Returns:
            Q: the updated Q-table.
            total_rewards: a list containing the rewards per episode.
    """
    total_rewards = []
    max_epsilon = epsilon
    for episode in range(episodes):
        state = env.reset()

        done = False
        rewards_episode = 0

        for step in range(max_steps):

            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            if done is True and reward == 0:
                reward = -1

            Q[state, action] = (Q[state, action] * (1 - alpha) +
                                alpha * (reward +
                                gamma * np.max(Q[new_state, :])))
            state = new_state
            rewards_episode += reward

            if done is True:
                break

        epsilon = (min_epsilon + (max_epsilon -
                   min_epsilon) * np.exp(-epsilon_decay * episode))

        total_rewards.append(rewards_episode)

    return Q, total_rewards
