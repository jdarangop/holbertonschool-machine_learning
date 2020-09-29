#!/usr/bin/env python3
""" Load the Environment """
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """ loads the pre-made FrozenLakeEnv evnironment from OpenAIs gym.
        Args:
            desc: (list/None) lists containing a custom description
                  of the map to load for the environment.
            map_name: (str/None) containing the pre-made map to load.
            is_slippery: (bool) boolean to determine if the ice is slippery.
        Returns:
            the environment.
    """
    return FrozenLakeEnv(desc, map_name, is_slippery)
