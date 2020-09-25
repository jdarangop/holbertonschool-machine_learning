#!/usr/bin/env python3
""" RNN Encoder """
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ class RNNEncoder """

    def __init__(self, vocab, embedding, units, batch):
        """ Initializer. """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       kernel_initializer="glorot_uniform",
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the
            RNN cell to a tensor of zeros.
            Args:
                None.
            Returns:
                (tf.Tensor) containing the initialized hidden states.
        """
        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """ 
            Args:
                x: (tf.Tensor) containing the input to the encoder layer
                   as word indices within the vocabulary.
                initial: (tf.Tensor) containing the initial hidden state.
            Returns:
                outputs: (tf.Tensor) containing the outputs of the encoder.
                hidden: (tf.Tensor) containing the last hidden state of the encoder.
        """
        outputs, hidden = self.gru(inputs=self.embedding(x), initial_state=initial)
        return outputs, hidden
