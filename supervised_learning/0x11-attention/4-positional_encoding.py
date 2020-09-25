#!/usr/bin/env python3
""" Positional Encoding """
import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer.
        Args:
            max_seq_len: (int) representing the maximum sequence length.
            dm: (int) the model depth.
        Returns:
            containing the positional encoding vectors.
    """
    result = np.zeros((max_seq_len, dm))
    for i in range(max_seq_len):
        tmp = []
        for j in range(int(dm / 2)):
            tmp.append(np.sin((i / (10000 ** (2 * j / dm)))))
            tmp.append(np.cos((i / (10000 ** (2 * j / dm)))))
        result[i, :] = np.array(tmp)

    return result
