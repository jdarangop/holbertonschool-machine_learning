#!/usr/bin/env python3
""" Transformer Decoder Block """
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """ DecoderBlock class """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """ Initializer.
            Args:
                dm: (int) the dimensionality of the model.
                h: (int) the number of heads.
                hidden: (int) the number of hidden units in the
                        fully connected layer.
                drop_rate: (float) the dropout rate.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(rate=drop_rate)

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
