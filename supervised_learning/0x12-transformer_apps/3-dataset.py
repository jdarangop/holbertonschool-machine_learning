#!/usr/bin/env python3
""" 3. Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset(object):
    """ Dataset class. """

    def __init__(self, batch_size, max_len):
        """ Initializer.
            Args:
                batch_size: (int) batch size for training/validation.
                max_len: (int) the maximum number of tokens
                         allowed per example sentence.
        """
        self.max_len = max_len
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        pt, en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt
        self.tokenizer_en = en
        padded_shapes = ([None], [None])
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_train = self.data_train.filter(self.f_max_len)
        self.data_train = \
            self.data_train.shuffle(metadata.splits['train'].num_examples)
        self.data_train = \
            self.data_train.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)
        self.data_train = \
            self.data_train.prefetch(tf.data.experimental.AUTOTUNE)
        self.data_valid = self.data_valid.map(self.tf_encode)
        self.data_valid = self.data_valid.filter(self.f_max_len)
        self.data_valid = \
            self.data_valid.padded_batch(batch_size,
                                         padded_shapes=padded_shapes)

    def tokenize_dataset(self, data):
        """ creates sub-word tokenizers for our dataset.
            Args:
                data: (tf.data.Dataset) whose examples are
                       formatted as a tuple (pt, en).
            Returns: tokenizer_pt, tokenizer_en
                tokenizer_pt is the Portuguese tokenizer
                tokenizer_en is the English tokenizer
        """
        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15)
        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ encodes a translation into tokens.
            Args:
                pt: (tf.Tensor) containing the Portuguese sentence.
                en: (tf.Tensor) containing the corresponding English sentence.
            Returns: pt_tokens, en_tokens
                pt_tokens: (tf.Tensor) containing the Portuguese tokens.
                en_tokens: (tf.Tensor) containing the English tokens.
        """
        pt_tokens = ([self.tokenizer_pt.vocab_size] +
                     self.tokenizer_pt.encode(pt.numpy()) +
                     [self.tokenizer_pt.vocab_size+1])
        en_tokens = ([self.tokenizer_en.vocab_size] +
                     self.tokenizer_en.encode(en.numpy()) +
                     [self.tokenizer_en.vocab_size+1])

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """ acts as a tensorflow wrapper for the encode instance method.
            Args:
                pt: (tf.Tensor) containing the Portuguese sentence.
                en: (tf.Tensor) containing the corresponding English sentence.
        """
        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en

    def f_max_len(self, x, y):
        """ Filter max_len. """
        return tf.logical_and(tf.size(x) <= self.max_len,
                              tf.size(y) <= self.max_len)
