#!/usr/bin/env python3
""" Cumulative N-gram BLEU score """
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


def cumulative_bleu(references, sentence, n):
    """ calculates the cumulative n-gram BLEU score for a sentence.
        Args:
            references: (list of list) with reference translations.
            sentence: (list) containing the model proposed sentence.
            n: (int) the size of the largest n-gram to use for evaluation.
        Returns:
            (float) the cumulative n-gram BLEU score..
    """
    bleus = []
    for k in range(1, n + 1):
        count = 0
        count_clip = 0
        for i in range(len(sentence) - k + 1):
            ngram = sentence[i:i+k]
            count += count_pattern(sentence, ngram)
            maximum = 0
            for reference in references:
                tmp = count_pattern(reference, ngram)
                if tmp > maximum:
                    maximum = tmp
            count_clip += maximum
        bleus.append(count_clip / count)

    len_MT = len(sentence)
    index = np.argmin([abs(len(i) - len_MT) for i in references])
    len_ref = len(references[index])
    if len_MT > len_ref:
        BP = 1
    else:
        BP = np.exp(1 - len_ref / len_MT)
    bleu_init = np.exp(np.sum(np.log(bleus) / n))
    bleu = BP * bleu_init

    return bleu
