#!/usr/bin/env python3
""" Create Masks """
import tensorflow as tf


def create_masks(inputs, target):
    """ creates all masks for training/validation.
        Args:
            inputs: (tf.Tensor) contains the input sentence.
            target: (tf.Tensor) contains the target sentence.
        Returns: encoder_mask, look_ahead_mask, decoder_mask
            encoder_mask: (tf.Tensor) padding mask to be applied
                          in the encoder.
            look_ahead_mask: (tf.Tensor) look ahead mask
                             to be applied in the decoder.
            decoder_mask: (tf.Tensor) padding mask to be applied
                          in the decoder.
    """
    seq_encoder = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    seq_decoder = tf.cast(tf.math.equal(target, 0), tf.float32)
    decoder_mask = target[:, tf.newaxis, tf.newaxis, :]
    mask = 1 - tf.linalg.band_part(tf.ones((target.shape[1],
                                            target.shape[1])), -1, 0)

    return encoder_mask, decoder_mask, mask
