#!/usr/bin/env python3
""" Create the loop """
import cmd
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


class QA_bot(cmd.Cmd):
    """ class QA_bot """

    prompt = 'Q: '

    def __init__(self, reference):
        """ Initializer. """
        super().__init__()
        self.reference = reference

    def question_answer(self, question):
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
        reference_tokens = tokenizer.tokenize(self.reference)
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

    def onecmd(self, arg):
        """ onecmd override """
        return cmd.Cmd.onecmd(self, arg.lower())

    def do_EOF(self, arg):
        """ default method. """
        print("A:")

    def default(self, arg):
        """ default method. """
        answer = self.question_answer(arg)
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


def answer_loop(reference):
    """ answers questions from a reference text.
        Args:
            reference: (str) the reference text.
        Returns:
            None.
    """
    QA_bot(reference).cmdloop()
