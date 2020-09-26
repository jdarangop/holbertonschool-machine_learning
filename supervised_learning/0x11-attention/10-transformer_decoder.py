#!/usr/bin/env python3
""" Transformer Decoder Block """
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """ DecoderBlock class """

    def __init__(self, N, dm, h, hidden, target_vocab,
                 max_seq_len, drop_rate=0.1):
        """ Initializer.
            Args:
                dm: (int) the dimensionality of the model.
                h: (int) the number of heads.
                hidden: (int) the number of hidden units in the
                        fully connected layer.
                drop_rate: (float) the dropout rate.
        """
        super().__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=target_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       self.dm)
        blocks = []
        for i in range(N):
            blocks.append(DecoderBlock(dm, h, hidden, drop_rate))
        self.blocks = blocks
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)

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
