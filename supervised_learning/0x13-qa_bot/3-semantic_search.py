#!/usr/bin/env python3
""" Semantic Search """
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import os


def semantic_search(corpus_path, sentence):
    """ performs semantic search on a corpus of documents.
        Args:
            corpus_path: (str) the path to the corpus of reference
            documents on which to perform semantic search.
            sentence: (str) the sentence from which to
                      perform semantic search.
        Returns:
            (str) the reference text of the document most
                  similar to sentence.
    """
    url = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    tokenizer = BertTokenizer.from_pretrained(url)
    model = TFAutoModelForQuestionAnswering.from_pretrained(url,
                                                            return_dict=True)

    filelist = os.listdir(corpus_path)
    maximo = None
    final_file = None
    for file in filelist:
        path_file = corpus_path + "/" + file
        if os.path.isfile(path_file):
            with open(path_file, 'rb') as f:
                reference = f.read().decode(errors='replace')

            inputs = tokenizer(sentence, reference,
                               add_special_tokens=True,
                               return_tensors="tf")
            input_ids = inputs["input_ids"].numpy()[0]
            text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            output = model(inputs)
            answer_start = tf.argmax(
                output.start_logits, axis=1
            ).numpy()[0]
            answer_end = (
                tf.argmax(output.end_logits, axis=1) + 1
            ).numpy()[0]
            first = output.start_logits[:, answer_start].numpy()[0]
            last = output.start_logits[:, answer_end].numpy()[0]

            if maximo is None:
                maximo = (first + last) / 2
                final_file = path_file
            elif (maximo < ((first + last) / 2)):
                maximo = (first + last) / 2
                final_file = path_file

    with open(final_file, 'rb') as f:
        result = f.read().decode(errors='replace')

    return result
