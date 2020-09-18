#!/usr/bin/env python3
""" N-gram BLEU score """
import numpy as np


def count_pattern(sentence, pattern):
    """ count the times a pattern is in a sentence.
        Args:
            sentence: (list) sentence to search.
            pattern (list) pattern to be found.
        Return:
            (int) number o f times the pattern is in the sentence.
    """
    n = len(pattern)
    counter = 0
    for i in range(len(sentence) - n + 1):
        if sentence[i:i+n] == pattern:
            counter += 1

    return counter


def ngram_bleu(references, sentence, n):
    """ calculates the n-gram BLEU score for a sentence.
        Args:
            references: (list of list) with reference translations.
            sentence: (list) containing the model proposed sentence.
            n: (int) the size of the n-gram to use for evaluation.
        Returns:
            (float) the n-gram BLEU score.
    """
    count = 0
    count_clip = 0
    for i in range(len(sentence) - n + 1):
        ngram = sentence[i:i+n]
        count += count_pattern(sentence, ngram)
        maximum = 0
        for reference in references:
            tmp = count_pattern(reference, ngram)
            if tmp > maximum:
                maximum = tmp
        count_clip += maximum

    len_MT = len(sentence)
    index = np.argmin([abs(len(i) - len_MT) for i in references])
    len_ref = len(references[index])
    if len_MT > len_ref:
        BP = 1
    else:
        BP = np.exp(1 - len_ref / len_MT)
    bleu_init = count_clip / count
    bleu = BP * bleu_init

    return bleu
