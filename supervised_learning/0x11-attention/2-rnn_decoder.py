#!/usr/bin/env python3
""" RNN Decoder """
import tensorflow as tf
import numpy as np
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ class RNNDecoder """

    def __init__(self, vocab, embedding, units, batch):
        """ Initializer.
            Args:
                vocab: (int) representing the size of the output vocabulary.
                embedding: (int) representing the dimensionality
                           of the embedding vector.
                units: (int) representing the number of hidden
                       units in the RNN cell.
                batch: (int)  representing the batch size.
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)
        self.F = tf.keras.layers.Dense(units=vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """ call method.
            Args:
                x: (tf.Tensor) containing the previous word in the target
                   sequence as an index of the target vocabulary.
                s_prev: (tf.Tensor) containing the previous
                        decoder hidden state.
                hidden_states: (tf.Tensor) containing the outputs
                               of the encoder.
            Returns:
                y: (tf.Tensor) containing the output word as a
                   one hot vector in the target vocabulary.
                s: (tf.Tensor) containing the new decoder hidden state.
        """
        emb = self.embedding(x)
        context, _ = self.attention(s_prev, hidden_states)
        cont_x = tf.concat([tf.expand_dims(context, 1), emb], axis=-1)
        out, s = self.gru(cont_x)
        out = tf.reshape(out, (out.shape[0], out.shape[2]))

        return self.F(out), s
