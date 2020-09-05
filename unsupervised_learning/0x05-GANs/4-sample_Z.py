#!/usr/bin/env python3
""" Sample Z """
import numpy as np


def sample_Z(m, n):
    """ creates input for the generator.
        Args:
            m: (int) the number of samples that should be generated.
            n: (int) the number of dimensions of each sample.
        Returns:
            Z: (numpy.ndarray) containing the uniform samples.
    """
    Z = np.random.uniform(low=-1, high=1, size=(m, n))
    return Z
