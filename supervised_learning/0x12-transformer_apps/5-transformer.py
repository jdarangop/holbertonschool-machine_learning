#!/usr/bin/env python3
""" Train """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
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


def sdp_attention(Q, K, V, mask=None):
    """ calculates the scaled dot product attention.
        Args:
            Q: (tf.Tensor) containing the query matrix.
            K: (tf.Tensor) containing the key matrix.
            V: (tf.Tensor) containing the value matrix.
            mask: (tf.Tensor) containing the optional mask,
                  or defaulted to None.
        Returns:
            output: (tf.Tensor) containing the scaled
                    dot product attention.
            weights: (tf.Tensor) containing the attention weights.
    """
    matmul = tf.linalg.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(Q)[-1], tf.float32)
    scale = matmul / tf.math.sqrt(dk)
    if mask is not None:
        scale += (mask * -1e9)
    attention_weights = tf.nn.softmax(scale, axis=-1)
    output = tf.linalg.matmul(attention_weights, V)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """ class MultiHeadAttention """

    def __init__(self, dm, h):
        """ Initializer.
            Args:
                dm: (int) representing the dimensionality
                    of the model.
                h: (int) representing the number of heads.
        """
        super().__init__()
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def call(self, Q, K, V, mask):
        """ call method.
            Args:
                Q: (tf.Tensor) containing the input to generate
                   the query matrix.
                K: (tf.Tensor) containing the input to generate
                   the key matrix.
                V: (tf.Tensor) containing the input to generate
                   the value matrix.
                mask: is always None.
            Returns:
                output: (tf.Tensor) containing the scaled dot
                        product attention.
                weights: (tf.Tensor) containing the attention weights.
        """
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        q = tf.reshape(q, (batch, -1, self.h, self.depth))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = self.Wq(K)
        k = tf.reshape(k, (batch, -1, self.h, self.depth))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = self.Wq(V)
        v = tf.reshape(v, (batch, -1, self.h, self.depth))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        out, weights = sdp_attention(q, k, v, mask)

        out = tf.transpose(out, perm=[0, 2, 1, 3])

        concat = tf.reshape(out, (batch, -1, self.dm))

        output = self.linear(concat)

        return out, weights


class EncoderBlock(tf.keras.layers.Layer):
    """ EncoderBlock class """

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
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate=drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=drop_rate)

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

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        """ call method.
            Args:
                x: (tf.Tensor) containing the input to the encoder block.
                training: (bool) to determine if the model is training.
                mask: the mask to be applied for multi head attention.
            Returns:
                (tf.Tensor) containing the blocks output.
        """
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.dense_hidden(out2d)
        ffn_output = self.dense_output(ffn_output)

        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(tf.keras.layers.Layer):
    """ DecoderBlock class """

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
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
        self.h = h
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(input_dim=input_vocab,
                                                   output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len,
                                                       self.dm)
        blocks = []
        for i in range(N):
            blocks.append(EncoderBlock(dm, h, hidden, drop_rate))
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
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


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

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask=None):
        """ call method.
            Args:
                x: (tf.Tensor) containing the input to the encoder block.
                training: (bool) to determine if the model is training.
                mask: the mask to be applied for multi head attention.
            Returns:
                (tf.Tensor) containing the blocks output.
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for i in range(self.N):
            x = self.blocks[i](x, enc_output, training,
                               look_ahead_mask, padding_mask)

        return x


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

    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        """ call method.
            Args:
                x: (tf.Tensor) containing the input to the encoder block.
                training: (bool) to determine if the model is training.
                mask: the mask to be applied for multi head attention.
            Returns:
                (tf.Tensor) containing the blocks output.
        """
        enc_output = self.encoder(inp, training, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.linear(dec_output)

        return final_output, attention_weights
