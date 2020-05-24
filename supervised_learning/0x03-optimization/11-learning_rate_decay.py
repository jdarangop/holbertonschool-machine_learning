#!/usr/bin/env python3
""" Learning Rate Decay """
import numpy as np


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ updates the learning rate using
        inverse time decay in numpy.
        alpha: (float) the initial learning rate.
        decay_rate: (int) the weight whice alpha will decay.
        global_step: (int) the number of passes of gradient
                     descent have elapsed.
        decay_step: (int)  the number of passes of gradient descent
                           that should occur before alpha is decayed.
        Returns: the updated alpha value.
    """
    alpha /= (1 + (global_step // decay_step))
    return alpha
