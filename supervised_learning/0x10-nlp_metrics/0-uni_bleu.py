#!/usr/bin/env python3
""" Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence.
        Args:
            references: (list of list) with reference translations.
            sentence: (list) containing the model proposed sentence.
        Returns:
            (float) the unigram BLEU score.
    """
    count = 0
    count_clip = 0
    for i in sentence:
        count += sentence.count(i)
        maximum = 0
        for reference in references:
            tmp = reference.count(i)
            if tmp > maximum:
                maximum = tmp
        # print(i, maximum)
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
