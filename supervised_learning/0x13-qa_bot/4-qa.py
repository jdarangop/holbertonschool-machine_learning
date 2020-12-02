#!/usr/bin/env python3
""" Multi-reference Question Answering """
import cmd
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, TFAutoModelForQuestionAnswering
import os


class QA_bot(cmd.Cmd):
    """ class QA_bot """

    prompt = 'Q: '

    def __init__(self, corpus_path):
        """ Initializer. """
        super().__init__()
        self.corpus_path = corpus_path

    def question_answer(question, reference):
        """ finds a snippet of text within a reference
            document to answer a question.
            Args:
                question: (str) containing the question to answer.
                reference: (str) containing the reference document
                        from which to find the answer.
            Returns:
                (str) containing the answer.
        """
        token_type = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        tokenizer = BertTokenizer.from_pretrained(token_type)
        model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

        question_tokens = tokenizer.tokenize(question)
        reference_tokens = tokenizer.tokenize(reference)
        tokens = (['[CLS]'] + question_tokens + ['[SEP]'] +
                  reference_tokens + ['[SEP]'])
        input_word_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_word_ids)
        input_ids = ([0] * (1 + len(question_tokens) + 1) +
                     [1] * (len(reference_tokens) + 1))

        input_word_ids, input_mask, input_ids = map(lambda t: tf.expand_dims(
            tf.convert_to_tensor(t, dtype=tf.int32), 0),
            (input_word_ids, input_mask, input_ids))
        outputs = model([input_word_ids, input_mask, input_ids])
        short_start = tf.argmax(outputs[0][0][1:]) + 1
        short_end = tf.argmax(outputs[1][0][1:]) + 1
        answer_tokens = tokens[short_start: short_end + 1]
        answer = tokenizer.convert_tokens_to_string(answer_tokens)

        if answer == "":
            return None

        return answer

    def semantic_search(self, sentence):
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
        model = TFAutoModelForQuestionAnswering.from_pretrained(
            url, return_dict=True)

        filelist = os.listdir(self.corpus_path)
        maximo = None
        final_file = None
        for file in filelist:
            path_file = self.corpus_path + "/" + file
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

    def onecmd(self, arg):
        """ onecmd override """
        return cmd.Cmd.onecmd(self, arg.lower())

    def do_EOF(self, arg):
        """ default method. """
        print("A:")

    def default(self, arg):
        """ default method. """
        reference = self.semantic_search(arg)
        answer = QA_bot.question_answer(arg, reference)
        if answer is None:
            answer = "Sorry, I do not understand your question."
        print("A:", answer)

    def do_exit(self, arg):
        """ exit method. """
        print("A:", "Goodbye")
        return True

    def do_quit(self, arg):
        """ quit method. """
        print("A:", "Goodbye")
        return True

    def do_goodbye(self, arg):
        """ goodbye method. """
        print("A:", "Goodbye")
        return True

    def do_bye(self, arg):
        """ bye method. """
        print("A:", "Goodbye")
        return True


def question_answer(corpus_path):
    """ answers questions from a reference text.
        Args:
            corpus_path: (str) the corpus_path.
        Returns:
            None.
    """
    QA_bot(corpus_path).cmdloop()
