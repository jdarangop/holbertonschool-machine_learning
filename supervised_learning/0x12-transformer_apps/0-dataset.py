#!/usr/bin/env python3
""" Dataset """
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset(object):
    """ Dataset class. """

    def __init__(self):
        """ Initializer.
            Args:
                None.
        """
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        self.data_train = examples['train']
        self.data_valid = examples['validation']
        pt, en = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt = pt
        self.tokenizer_en = en

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
