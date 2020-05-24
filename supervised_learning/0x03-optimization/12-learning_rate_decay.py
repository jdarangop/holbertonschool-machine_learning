#!/usr/bin/env python3
""" Learning Rate Decay """
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate,
                        global_step, decay_step):
    """ creates a learning rate decay operation
        in tensorflow using inverse time decay.
        alpha: (float) the initial learning rate.
        decay_rate: (int) the weight whice alpha will decay.
        global_step: (int) the number of passes of gradient
                     descent have elapsed.
        decay_step: (int)  the number of passes of gradient descent
                           that should occur before alpha is decayed.
        Returns: the learning rate decay operation.
    """
    return tf.train.inverse_time_decay(learning_rate=alpha,
                                       global_step=global_step,
                                       decay_steps=decay_step,
                                       decay_rate=decay_rate,
                                       staircase=True)
