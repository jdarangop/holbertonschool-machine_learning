#!/usr/bin/env python3
""" Bag Of Words """
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """ creates a bag of words embedding matrix.
        Args:
            sentences: (list) containing the sentences to analyze.
            vocab: (list)
        Returns:
            embeddings: (numpy.ndarray) containing the embeddings.
            features: (list) the features used for embeddings.
    """
    vectorizer = CountVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    features = vectorizer.get_feature_names()
    return X.toarray(), features
