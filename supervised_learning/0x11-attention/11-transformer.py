#!/usr/bin/env python3
""" Transformer Decoder Block """
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """ DecoderBlock class """

    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ Initializer.
            Args:
                dm: (int) the dimensionality of the model.
                h: (int) the number of heads.
                hidden: (int) the number of hidden units in the
                        fully connected layer.
                drop_rate: (float) the dropout rate.
        """
        super().__init__()
        self.encoder = Encoder(N, dm, h, hidden, input_vocab,
                               max_seq_input, drop_rate)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab,
                               max_seq_input, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, x, training, mask=None):
        """ call method.
            Args:
                x: (tf.Tensor) containing the input to the encoder block.
                training: (bool) to determine if the model is training.
                mask: the mask to be applied for multi head attention.
            Returns:
                (tf.Tensor) containing the blocks output.
        """
        out1,  _ = self.mha(x, x, x, mask)
        out1 = self.dropout1(out1, training=training)
        out1 = self.layernorm1(x + out1)

        hidden_out = self.dense_hidden(out1)
        out2 = self.dense_output(hidden_out)
        out2 = self.dropout2(out2, training=training)
        output = self.layernorm2(out1 + out2)

        return output
